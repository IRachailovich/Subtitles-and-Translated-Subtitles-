# SubGen Partner Setup and Use Guide

This guide explains the three SubGen delivery forms and how a partner obtains, configures, runs, and updates each one.

## A. Choose the correct delivery form

| Delivery form | Included interfaces | Installation model | Internet hosting |
| --- | --- | --- | --- |
| Source application | CLI and local browser UI | Clone the repository and install its Python dependencies | No; both run on the partner's computer |
| Frozen Windows desktop application | Installed desktop UI and private-LAN phone access | Download the `.exe` installer from GitHub Releases | No; phone access connects to the Windows computer |
| Cloud web/mobile application | Hosted browser UI and installable PWA | Deploy the repository to separately configured cloud services | Yes |

The source CLI and local browser UI share the same core pipeline. The Windows installer packages that local application for non-developers. The cloud application is a separate hosted operating form and requires infrastructure; cloning alone does not deploy it.

## B. Source application: clone and install

Use this setup for both the CLI and local browser UI.

### B1. Requirements

- Git, or a ZIP downloaded from GitHub
- Python 3.10 or newer; Python 3.11 is recommended
- Conda, `venv`, or another isolated Python environment
- FFmpeg and ffprobe on `PATH`
- Free disk space for dependencies, models, temporary audio, outputs, and optional burned videos
- Provider credentials for any selected API provider

A compatible GPU is optional. CPU execution works but local transcription and alignment can be much slower.

### B2. Clone into any chosen folder

Open a terminal in the parent folder where the repository should be created:

```text
git clone https://github.com/CarlFriGauss/Subtitles-and-Translated-Subtitles-.git
cd Subtitles-and-Translated-Subtitles-
```

There is no required drive letter or absolute installation path. In all commands below, `.` means the repository directory currently open in the terminal.

Partners without Git can use **Code → Download ZIP** on GitHub, extract it into any folder, and open a terminal in the extracted folder. Git is recommended because it supports controlled updates and exact version identification.

### B3. Create an environment with Conda

```text
conda create -n subgen python=3.11 -y
conda activate subgen
conda install -c conda-forge ffmpeg -y
python -m pip install -e ".[align]"
```

`.[align]` installs the normal application plus the optional local transcription and WhisperX alignment stack. Use `python -m pip install -e .` only when the smaller base installation is intentional.

### B4. Create an environment with `venv`

Install [FFmpeg](https://ffmpeg.org/download.html) separately for the operating system, then create the Python environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[align]"
```

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[align]"
```

### B5. Verify the tools

```text
python --version
ffmpeg -version
ffprobe -version
SubGen --help
```

If `SubGen` is not found, confirm that the intended environment is activated and repeat `python -m pip install -e ".[align]"` from the repository directory.

### B6. Store private data outside the clone

Without an override, source mode stores configuration and application state inside the repository directory. A separate private data directory is recommended.

Windows PowerShell:

```powershell
$env:SUBGEN_DATA_DIR = Join-Path $HOME ".subgen"
```

macOS or Linux:

```bash
export SUBGEN_DATA_DIR="$HOME/.subgen"
```

Set the variable again in each new terminal session or configure it persistently through the operating system. Do not share or commit the selected directory.

### B7. Configure providers

```text
SubGen setup
```

Setup chooses separate transcription and translation providers and stores provider profiles. Useful follow-up commands are:

```text
SubGen providers
SubGen languages
SubGen config
```

Provider API calls can cost money. The account owning each key is responsible for its provider billing. Local models avoid provider API charges but may download large model files and consume significant CPU/GPU, RAM, and disk.

## C. Run the CLI

### C1. Interactive run

```text
SubGen
```

or:

```text
SubGen run
```

SubGen asks for missing values interactively.

### C2. Command-line run

Replace every capitalized placeholder with a path or value appropriate for the current computer:

```text
SubGen run --video "PATH_TO_VIDEO" --output-dir "PATH_TO_OUTPUT_FOLDER" --target-language en --transcription-provider google --translation-provider openai
```

Original-language subtitles without translation or burning:

```text
SubGen run --video "PATH_TO_VIDEO" --output-dir "PATH_TO_OUTPUT_FOLDER" --target-language none --transcription-provider local --model-size medium --no-burn
```

Show all supported options:

```text
SubGen run --help
```

### C3. Batch processing

`--input` accepts a video file, directory, ZIP file, or text/list file containing video paths.

```text
SubGen batch --input "PATH_TO_BATCH_INPUT" --output-dir "PATH_TO_BATCH_OUTPUT" --target-language en --transcription-provider google --translation-provider openai --continue-on-error --no-prompts
```

Check or stop a batch gracefully:

```text
SubGen batch-status --output-dir "PATH_TO_BATCH_OUTPUT"
SubGen batch-stop --output-dir "PATH_TO_BATCH_OUTPUT"
```

Rerun `SubGen batch` with the same output directory to resume saved batch state. Use `SubGen batch --help` for the exact options in the checked-out version.

### C4. Review and output

Depending on the selected options, SubGen produces source subtitles, translated subtitles, review state, manifests, cost records, and optionally a burned video.

```text
SubGen review list
SubGen costs --output-dir "PATH_TO_OUTPUT_FOLDER"
SubGen balance --output-dir "PATH_TO_OUTPUT_FOLDER"
```

Review the complete transcript, timestamps, repetitions, mixed-language passages, translation, and reported issues. Editing a reviewed draft invalidates its previous approval; approve the exact corrected draft before burning.

### C5. Long jobs

- Keep automatic API transcription chunking enabled unless a specific provider requires otherwise.
- Use a stable output directory so manifests, checkpoints, reviews, and accepted intermediate results remain available.
- Keep ample free disk space and prevent the computer from sleeping.
- Do not delete the output or data directory while processing is active.
- Provider upload, request, duration, quota, model, and billing limits still apply.

## D. Run the local browser UI

Activate the same source environment used for the CLI, set `SUBGEN_DATA_DIR` if desired, and run:

```text
python run_web.py
```

The default browser opens at:

```text
http://localhost:8080
```

The default host is `127.0.0.1`, so the service is available only on that computer.

Normal workflow:

1. Open **API Settings** and save the provider profiles required for the job.
2. Open **Upload & Config** and select a video and output/cache directory.
3. Choose source language or automatic detection, target language, providers, models, alignment mode, and subtitle options.
4. Start processing and follow **Process & Logs**.
5. Inspect and correct the complete result in **Review & Edit**.
6. Approve the exact final draft.
7. Start the separate burn stage only if a rendered video is required.
8. Export the video and subtitle files.

Keep the terminal open. Press `Ctrl+C` to stop the server. Do not expose the local development server directly to the internet.

## E. Install the frozen Windows desktop application

The desktop installer is the simplest choice for a partner who does not need source development.

### E1. Download and verify

1. Open the [SubGen v0.4.1 release](https://github.com/CarlFriGauss/Subtitles-and-Translated-Subtitles-/releases/tag/v0.4.1).
2. Download `SubGen_0.4.1_x64-setup.exe`.
3. Calculate its SHA-256:

```powershell
Get-FileHash "$HOME\Downloads\SubGen_0.4.1_x64-setup.exe" -Algorithm SHA256
```

Expected value:

```text
57DADD5AF9044CFE63BD12641D7033B8626687CFCD48A65EBC123136F61C2234
```

If the hash differs, delete the file and download it again from the release page.

### E2. Install

1. Double-click the verified installer.
2. Because the current installer is not digitally signed, Windows SmartScreen may warn. Use **More info → Run anyway** only after verifying the release source and SHA-256.
3. Launch **SubGen** from the Start menu or installed shortcut.

The installer is per-user and includes the frozen Python backend, desktop shell, web assets, FFmpeg/ffprobe, fonts, and alignment dependencies. Python, Conda, Rust, Node.js, and a separate FFmpeg installation are not required.

Application data is stored under `%LOCALAPPDATA%\SubGen`. Treat that directory as private.

### E3. Use

1. Configure provider keys in **API Settings**.
2. Select the video and output/cache directory in **Upload & Config**.
3. Choose languages, providers, models, timing, and subtitle settings.
4. Monitor **Process & Logs**.
5. Correct and approve the exact final draft in **Review & Edit**.
6. Burn only the approved draft when needed.
7. Export the final video and subtitle files.

The first local-model operation may download model files. API calls may incur provider charges. Keep the computer awake for long transcription or burn stages.

## F. Use a phone with the installed desktop application

This is private-LAN access to the Windows application, not cloud hosting.

1. Connect the Windows computer and phone to the same trusted private network.
2. Start SubGen on Windows and open **Mobile Access**.
3. Scan the displayed QR code with the phone.
4. If Windows Firewall asks, allow access on private networks only.
5. Approve local-network access on the phone if the operating system requests it.
6. Keep the Windows application running while using the phone interface.

Treat the QR/pairing URL as a secret. Do not configure router port forwarding, expose port `8765` to the internet, or use this feature on an untrusted network. A VPN must allow LAN traffic on both devices.

## G. Deploy the future cloud web/mobile PWA

The repository contains cloud source code, but GitHub access alone is not a deployment. The deployment owner needs separate service accounts, security configuration, operational monitoring, and an accepted cost model.

### G1. Clone the approved source

```text
git clone https://github.com/CarlFriGauss/Subtitles-and-Translated-Subtitles-.git
cd Subtitles-and-Translated-Subtitles-
git switch main
```

For a fixed release snapshot:

```text
git fetch --tags
git switch --detach v0.4.1
```

### G2. Required services

- Vercel for the static PWA
- Supabase for authentication and PostgreSQL
- Cloudflare R2 for private media and artifact storage
- Google Cloud Run for the API and one-shot processing workers
- Google Secret Manager for deployment secrets
- User-selected transcription and translation providers

Read [cloud-webapp.md](cloud-webapp.md) completely before creating resources or entering credentials.

### G3. Safe deployment order

1. Use a dedicated Google Cloud project.
2. Configure Supabase authentication, database access, and allowed redirect URLs.
3. Create a private R2 bucket, bucket-scoped credentials, CORS rules, and retention lifecycle.
4. Deploy the API and worker with processing disabled.
5. Install and test the documented cost kill switch.
6. Configure Vercel with the final API origin.
7. Test authentication, account isolation, credential encryption, upload authorization, job creation, review, approval invalidation, download authorization, and retention.
8. Review quotas, timeouts, CPU/RAM, temporary disk, storage bandwidth, provider limits, monitoring, backups, incident response, privacy, and expected costs.
9. Enable processing only after those checks pass.

No deployment can promise unlimited video processing at zero cost. Provider limits and infrastructure resources remain real constraints.

### G4. Install the PWA after deployment

- Chrome or Edge on desktop/Android: use the browser's **Install app** command.
- Safari on iPhone/iPad: use **Share → Add to Home Screen**.

The installed PWA remains dependent on the deployed internet services. It is not an offline transcription engine and is not a native App Store or Google Play application.

## H. Update and support

Update a source clone:

```text
git switch main
git pull --ff-only
python -m pip install -e ".[align]"
```

Before reporting a problem, collect:

- product form: CLI, local browser, desktop installer, desktop phone access, or cloud PWA;
- operating system;
- Git commit/tag or installer version;
- provider and model selections;
- media duration and format;
- exact command or UI action;
- exact error and relevant non-secret logs.

Never attach API keys, OAuth tokens, database URLs, cloud secrets, private videos, transcripts, subtitles, or credential files to a public GitHub issue.
