# SafeSpeak

Audio-free workplace speech monitoring via Visual Speech Recognition on CCTV footage.

## Project structure

```
safespeak/
├── main.py                  ← launch all three workers
├── workers/
│   ├── worker1.py           ← camera → face tracking → lip ROI → disk
│   ├── worker2.py           ← disk frames → VSR → transcript
│   ├── worker3.py           ← transcript → keyword flag → report
│   └── vsr_model.py         ← # TO-DO
├── config/
│   └── keywords.txt         ← one keyword/phrase per line
├── dashboard/
│   └── index.html           ← open in browser, no server needed
├── inputs/                  ← {face_id}/{n}.png (auto-managed)
└── outputs/                 ← transcripts + flagging reports
```

## Quick start

```bash
pip install -r requirements.txt

python main.py
```

Open `dashboard/index.html` in a browser to manage employee/face mappings and view reports.

## Keywords

Edit `config/keywords.txt` at any time — Worker 3 picks up changes immediately.
Lines starting with `#` are comments. Matching is case-insensitive and word-boundary aware.

## Reports

Daily report: `outputs/report_DDMMYYYY.txt`
Transcripts:  `outputs/{face_id}_{DDMMYYYY:HHMM}.txt`
