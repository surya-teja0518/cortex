# CORTEX on Windows

The daily pipeline runs in the cloud (GitHub Actions) and the dashboard is online
(https://cortex-surya.vercel.app), so a Windows PC only needs CORTEX for the **local** jobs: tailoring and
building packets from your resume files, which never leave your computers. The full test suite runs on
Windows in CI on every change.

## 1. Install (once, ~10 min)

1. **Python 3.12** from python.org. In the installer, tick **"Add python.exe to PATH"**.
2. **Git for Windows** from git-scm.com (defaults are fine).
3. Open **PowerShell** and clone into your **user folder**, not into a OneDrive-synced folder:
   ```powershell
   cd $HOME
   git clone https://github.com/surya-teja0518/cortex.git
   cd cortex
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -e ".[dev,web]"
   setx PYTHONUTF8 1
   ```
   - Why not Desktop or Documents: on many Windows PCs those are synced to OneDrive, which would upload your
     `.env` and resume files to the cloud. `$HOME\cortex` is not synced.
   - `PYTHONUTF8=1` lets the console print job titles with non-English characters. Open a new PowerShell
     window after `setx`.
   - If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.
   - No Node.js, Docker or Ollama needed. The default tailor mode sends your own resume PDFs.

## 2. Copy your private files from the Mac

Copy these from the Mac's `cortex` folder into `$HOME\cortex` on Windows, **by USB stick or a password
manager**. Never by email, chat or an unencrypted cloud drive:

| File | Why |
|---|---|
| `.env` | database URL + LLM keys (drop the `GMAIL_*` and `VERCEL_TOKEN` lines: not needed here) |
| `config.local.yaml` | your personal settings (resume file names) |
| `master_resume.yaml` | Truth Guard and the Workday sheet |
| `contact.yaml`, `answers.yaml` | packet header and standard answers |
| `resumes\*.pdf` | your three resumes (the packet copies them) |

Not needed on Windows: `gmail-token.json`, `google-service-account.json`, `skills_profile.json`, `data\`.
Gmail, sheet sync and scoring all run in the cloud job.

Windows ignores Unix file permissions (`chmod 600`). Your user folder's permissions keep these files
private to your Windows account, which is why the repo must live under `$HOME`.

## 3. Check it works

```powershell
cortex report          # your applications, from the shared Neon database
cortex llm-check       # Groq / Gemini reachable (Ollama shows as down: fine)
```

## 4. What you do on Windows

- `cortex packet <job_id>`: build the PDF packet for a job you approved on the phone.
- `cortex tailor <job_id>`: prepare a resume for a job (then approve it in the dashboard).
- `cortex web`: the dashboard on http://127.0.0.1:8000 (optional; the online one does the same).

## Do NOT run on Windows

- `cortex daily`, `cortex schedule`: the cloud job already does the daily run. A second scheduler would run
  everything twice.
- `cortex migrate-db`: the data is already in Neon. Copying again is refused anyway.
