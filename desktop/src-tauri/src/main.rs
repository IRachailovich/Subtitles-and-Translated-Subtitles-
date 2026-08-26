#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use tauri::Manager;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

struct BackendProcess(Mutex<Option<Child>>);

const BACKEND_ADDRESS: &str = "127.0.0.1:8765";
const BACKEND_URL: &str = "http://127.0.0.1:8765";
const STARTUP_ATTEMPTS: usize = 600;
const STARTUP_RETRY_DELAY: Duration = Duration::from_secs(1);

fn backend_http_is_ready(address: SocketAddr) -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(500)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(750)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(750)));
    if stream
        .write_all(b"GET /api/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }

    let mut response = Vec::with_capacity(512);
    if stream.read_to_end(&mut response).is_err() {
        return false;
    }
    http_response_is_ok(&response)
}

fn http_response_is_ok(response: &[u8]) -> bool {
    response.starts_with(b"HTTP/1.1 200") || response.starts_with(b"HTTP/1.0 200")
}

fn window_is_on_backend(window: &tauri::WebviewWindow) -> bool {
    window
        .url()
        .map(|url| url.as_str().starts_with(BACKEND_URL))
        .unwrap_or(false)
}

fn connect_window_to_backend(handle: tauri::AppHandle) {
    let address: SocketAddr = BACKEND_ADDRESS.parse().unwrap();
    let backend_url = tauri::Url::parse(BACKEND_URL).unwrap();

    for _ in 0..STARTUP_ATTEMPTS {
        let Some(window) = handle.get_webview_window("main") else {
            return;
        };
        if window_is_on_backend(&window) {
            return;
        }
        if backend_http_is_ready(address) {
            let _ = window.navigate(backend_url.clone());
        }
        std::thread::sleep(STARTUP_RETRY_DELAY);
    }

    if let Some(window) = handle.get_webview_window("main") {
        let _ = window.eval(
            "document.querySelector('p').textContent = 'The subtitle engine could not start. Close SubGen and try again.'; document.querySelector('.bar').style.animation = 'none';",
        );
    }
}

fn backend_executable(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    if let Ok(configured) = std::env::var("SUBGEN_BACKEND_EXE") {
        return Ok(PathBuf::from(configured));
    }
    let resources = app
        .path()
        .resource_dir()
        .map_err(|error| error.to_string())?;
    Ok(resources
        .join("resources")
        .join("backend")
        .join("SubGenBackend.exe"))
}

fn spawn_backend(app: &tauri::AppHandle) -> Result<Child, String> {
    let data_dir = app
        .path()
        .app_local_data_dir()
        .map_err(|error| error.to_string())?;
    fs::create_dir_all(&data_dir).map_err(|error| error.to_string())?;
    let log_path = data_dir.join("backend.log");
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| error.to_string())?;
    let stderr = stdout.try_clone().map_err(|error| error.to_string())?;
    let resources = app
        .path()
        .resource_dir()
        .map_err(|error| error.to_string())?;
    let ffmpeg_dir = resources.join("resources").join("ffmpeg");
    let existing_path = std::env::var_os("PATH").unwrap_or_default();
    let joined_path = std::env::join_paths(
        std::iter::once(ffmpeg_dir).chain(std::env::split_paths(&existing_path)),
    )
    .map_err(|error| error.to_string())?;

    let mut command = Command::new(backend_executable(app)?);
    command
        .args(["--host", "0.0.0.0", "--port", "8765", "--lan"])
        .env("SUBGEN_DATA_DIR", &data_dir)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .env("PATH", joined_path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    #[cfg(windows)]
    command.creation_flags(0x08000000);
    command.spawn().map_err(|error| error.to_string())
}

fn cleanup_video_sessions(app: &tauri::AppHandle) {
    if let Ok(data_dir) = app.path().app_local_data_dir() {
        let _ = fs::remove_dir_all(data_dir.join("uploads").join("jobs"));
    }
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(BackendProcess(Mutex::new(None)))
        .setup(|app| {
            let child = spawn_backend(app.handle())?;
            *app.state::<BackendProcess>().0.lock().unwrap() = Some(child);

            let handle = app.handle().clone();
            std::thread::spawn(move || connect_window_to_backend(handle));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build SubGen desktop application");

    app.run(|app_handle, event| {
        if matches!(
            event,
            tauri::RunEvent::Exit | tauri::RunEvent::ExitRequested { .. }
        ) {
            if let Some(mut child) = app_handle
                .state::<BackendProcess>()
                .0
                .lock()
                .unwrap()
                .take()
            {
                let _ = child.kill();
                let _ = child.wait();
            }
            cleanup_video_sessions(app_handle);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::http_response_is_ok;

    #[test]
    fn accepts_only_successful_http_health_responses() {
        assert!(http_response_is_ok(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"
        ));
        assert!(http_response_is_ok(b"HTTP/1.0 200 OK\r\n\r\n"));
        assert!(!http_response_is_ok(
            b"HTTP/1.1 503 Service Unavailable\r\n\r\n"
        ));
        assert!(!http_response_is_ok(b""));
    }
}
