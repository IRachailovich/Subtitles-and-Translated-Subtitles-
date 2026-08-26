# SubGen

SubGen transcribes speech, aligns timestamps, translates subtitles, supports review and correction, writes subtitle files, and can burn approved subtitles into video.

## Choose how to run SubGen

| Delivery form | Interfaces | Clone required? | Best for |
| --- | --- | --- | --- |
| Source application | CLI and local browser UI | Yes | Developers, automation, batch work, and running locally from Python |
| Windows desktop application | Installed desktop UI and optional phone access over a private LAN | No | Partners who want a normal Windows installer |
| Cloud application | Hosted web UI and installable mobile/desktop PWA | Yes, for the deployment owner | A future managed multi-user internet deployment |

The cloud PWA and the desktop phone-access feature are different products. Desktop phone access connects to one Windows computer on the same trusted network. The cloud PWA requires separately deployed internet services.

For complete partner instructions, see [docs/PARTNER_GUIDE.md](docs/PARTNER_GUIDE.md).

## 1. CLI or local browser app from source

### Requirements

- Git
- Python 3.10 or newer; Python 3.11 is recommended
- Conda, `venv`, or another isolated Python environment
- FFmpeg and ffprobe available on `PATH`
- Provider credentials if using paid/API providers
- Sufficient disk space for media, outputs, temporary audio, and optional local models

Clone the repository into any folder you choose:

```text
git clone https://github.com/CarlFriGauss/Subtitles-and-Translated-Subtitles-.git
cd Subtitles-and-Translated-Subtitles-
```

The remaining commands are run from that repository directory. No fixed drive letter or installation directory is required.

### Conda setup

```text
conda create -n subgen python=3.11 -y
conda activate subgen
conda install -c conda-forge ffmpeg -y
python -m pip install -e ".[align]"
```

The `align` extra installs local transcription and WhisperX alignment dependencies. For a smaller base installation without those optional components, use `python -m pip install -e .`.

Verify the environment:

```text
python --version
ffmpeg -version
ffprobe -version
SubGen --help
```

### `venv` setup instead of Conda

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

### Keep user data outside the clone

Source mode stores configuration and local state in the repository directory unless `SUBGEN_DATA_DIR` is set. A separate private data directory is recommended.

Windows PowerShell:

```powershell
$env:SUBGEN_DATA_DIR = Join-Path $HOME ".subgen"
```

macOS or Linux:

```bash
export SUBGEN_DATA_DIR="$HOME/.subgen"
```

Do not commit or share this directory. It may contain configuration, database state, OAuth files, model caches, uploads, and credential metadata.

### Configure providers

```text
SubGen setup
SubGen providers
SubGen config
```

Provider calls may incur charges. Local models avoid provider API charges but use local CPU/GPU, RAM, disk, and model downloads.

### Run the CLI

Interactive mode:

```text
SubGen
```

Command example—replace the capitalized placeholders with paths on the current computer:

```text
SubGen run --video "PATH_TO_VIDEO" --output-dir "PATH_TO_OUTPUT_FOLDER" --target-language en --transcription-provider google --translation-provider openai
```

Original-language SRT without translation or video burning:

```text
SubGen run --video "PATH_TO_VIDEO" --output-dir "PATH_TO_OUTPUT_FOLDER" --target-language none --transcription-provider local --no-burn
```

Run `SubGen run --help` or `SubGen batch --help` for all options.

### Run the local browser application

```text
python run_web.py
```

SubGen opens the default browser at <http://localhost:8080>. The server listens on `127.0.0.1` by default, so it is local to that computer. Keep the terminal open while using the application and press `Ctrl+C` to stop it.

## 2. Windows desktop installer

Cloning the repository is not required.

1. Open the [SubGen v0.4.1 release](https://github.com/CarlFriGauss/Subtitles-and-Translated-Subtitles-/releases/tag/v0.4.1).
2. Download `SubGen_0.4.1_x64-setup.exe`.
3. Verify its SHA-256 before running it.

PowerShell:

```powershell
Get-FileHash "$HOME\Downloads\SubGen_0.4.1_x64-setup.exe" -Algorithm SHA256
```

Expected SHA-256:

```text
57DADD5AF9044CFE63BD12641D7033B8626687CFCD48A65EBC123136F61C2234
```

The installer is not digitally signed, so Windows SmartScreen may warn. Continue only when the installer came from the release above and the hash matches.

The desktop installer contains the application backend, web UI, FFmpeg/ffprobe, fonts, and alignment dependencies. End users do not need Python, Conda, Rust, Node.js, or a separate FFmpeg installation.

Installed application data is stored below `%LOCALAPPDATA%\SubGen`.

### Phone access through the desktop app

Open **Mobile Access** in the installed app, connect the phone and computer to the same trusted private network, and scan the displayed QR code. Allow Windows Firewall access on private networks only. Never expose SubGen through router port forwarding or an untrusted Wi-Fi network.

## 3. Future cloud web/mobile PWA

Cloning the repository provides the source code but does not create a working cloud service. A deployment owner must configure Supabase, Cloudflare R2, Google Cloud, Vercel, authentication, encryption secrets, quotas, monitoring, retention, and billing controls.

```text
git clone https://github.com/CarlFriGauss/Subtitles-and-Translated-Subtitles-.git
cd Subtitles-and-Translated-Subtitles-
```

Then follow [docs/cloud-webapp.md](docs/cloud-webapp.md) completely. Keep processing disabled until authentication, account isolation, storage authorization, uploads, review, downloads, limits, monitoring, and cost controls have been tested.

The cloud client is an installable PWA, not a native App Store or Google Play application. A separate native mobile project would be required for store distribution.

## Updating a source clone

```text
git switch main
git pull --ff-only
python -m pip install -e ".[align]"
```

To run the exact public source associated with the installer release:

```text
git fetch --tags
git switch --detach v0.4.1
```

## Security and support

- Never commit API keys, OAuth tokens, database URLs, cloud secrets, videos, transcripts, subtitles, or generated artifacts.
- Keep original media until outputs have been reviewed and backed up.
- Automated transcription, alignment, speaker handling, mixed-language processing, and translation can still make mistakes; review subtitles before publication.
- When reporting a problem, include the product form, operating system, Git commit/tag or installer version, provider/model, media duration/format, and exact error. Do not attach credentials or private media to a public issue.
