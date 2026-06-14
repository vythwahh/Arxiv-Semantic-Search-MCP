# WhaleBeaconNet

> *When whales can't hear the ships coming, we listen for them.*

Real-time passive acoustic detection system to reduce vessel strike risk for endangered cetaceans in high-traffic Arctic shipping corridors.

---

## Motivation

Arctic shipping traffic is expanding rapidly as climate change opens new maritime routes — directly overlapping with the habitat of bowhead whales (*Balaena mysticetus*), one of the most endangered cetacean populations on Earth. Unlike human-generated vessel noise, whale vocalizations carry rich bioacoustic signatures that can be detected, classified, and acted upon in real time.

Each cetacean species communicates at distinct, non-overlapping frequency ranges — a biological constraint that makes acoustic classification both feasible and reliable:

| Species | Vocalization Type | Frequency Range |
|---------|------------------|-----------------|
| Blue whale | Low-frequency moans | 10–40 Hz |
| Fin whale | 20 Hz pulses | ~20 Hz |
| Humpback whale | Complex songs | 20 Hz – 4 kHz |
| Sperm whale | Echolocation clicks | 2–30 kHz |
| Bowhead whale | Variable songs, calls | 25–500 Hz |

Vessel noise, while broadband, lacks the structured biological patterns of cetacean vocalizations — making species-level classification robust even in noisy maritime environments.

WhaleBeaconNet listens so they don't have to.

---

## Technical Pipeline

```
Raw hydrophone audio (.wav / .opus)
        |
        v
[1. DE Layer] Audio ingestion + chunking (3-5s segments)
        |
        v
[2. DS Layer] Denoising + Mel-Spectrogram feature extraction (Librosa)
        |
        v
[3. ML Layer] BirdNET transfer learning (fine-tuned for cetacean bioacoustics)
             Multi-class: fin whale | humpback | sperm whale | background
        |
        v
[4. Inference] Confidence thresholding (>= 0.75 to trigger alert)
               Low confidence -> flag for human review
        |
        v
[5. Integration] Cross-reference with AIS vessel traffic data
                 AnomalyAIS engine detects dark vessels in same zone
        |
        v
[6. Output] Risk score per corridor + Streamlit dashboard
```

---

## Why BirdNET

BirdNET (Cornell Lab of Ornithology) is chosen over general-purpose audio models (YAMNet, PANNs) for a specific reason: it was designed for wildlife bioacoustics in noisy natural environments.

Both bird vocalizations and cetacean calls share structural properties — whistles, frequency-modulated sweeps, and patterned calls against a background of environmental noise (wind, rain, waves). BirdNET's convolutional layers are optimized to extract biological frequency patterns from spectrograms while suppressing ambient noise — directly transferable to underwater hydrophone recordings where vessel engine rumble and wave noise are the dominant interference.

Known limitation: BirdNET is optimized for ~100 Hz–15 kHz. The lowest bowhead infrasound components (~25 Hz) fall below this range and may not be captured.

---

## Dataset

| Source | Species | Type |
|--------|---------|------|
| Kaggle — NOAA Fisheries | Humpback, fin, sperm, minke | Public domain, wav |
| DCLDE 2015 Workshop | Fin whale, blue whale | Annotated, research-grade |
| AWI OPUS Portal | Arctic soundscape (Svalbard) | Raw hydrophone, unlabeled |
| UW Bowhead Songs (Fram Strait) | Bowhead whale | 184 labeled songs, CC-BY |

Primary conservation focus: **Bowhead whale** (*Balaena mysticetus*) — Arctic population critically at risk from expanding shipping lanes driven by climate change.

Note: Labeled bowhead data is scarce relative to other cetacean species. The model is trained on species with larger annotated datasets and is designed for extension to bowhead as labeled Arctic data becomes available. Raw AWI hydrophone recordings from Svalbard (Lat 76°N) are used for pipeline validation and visualization.

---

## AIS Integration

Whale presence detections are cross-referenced with vessel AIS data using the [AnomalyAIS](https://github.com/vythwahh/AnomalyAIS) engine — a deterministic state machine that classifies vessels into OK, LOST, and MONITORING states based on AIS ping gaps.

A double risk flag is raised when:
- Whale vocalization is detected in a zone, AND
- A vessel in that zone is in LOST or MONITORING state (AIS dark)

This directly addresses the real-world limitation that vessels operating recklessly often disable AIS transponders — making whale-side detection the only reliable signal.

---

## Streamlit Dashboard

| Tab | Features |
|-----|---------|
| Detection | Upload `.wav` / `.opus` → spectrogram → species prediction + confidence |
| GIS Map | Arctic shipping lane polygons (GeoJSON) overlaid with whale detection zones → risk score per corridor |
| Detection Log | History table: timestamp, species, confidence, risk level → CSV export |
| AI Assistant | RAG-powered chatbox (Groq/Llama 3.1) grounded in cetacean acoustic profiles and migration ecology — answers species presence queries based on location, season, and environmental context |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Audio processing | Librosa, NumPy, SciPy |
| Feature extraction | Mel-Spectrogram, MFCCs |
| Transfer learning | BirdNET (Cornell Lab), PyTorch |
| AIS integration | AnomalyAIS engine |
| Geospatial | Folium, GeoJSON (IMO/GFW shipping lanes) |
| AI Assistant | Groq API, Llama 3.1, RAG pipeline |
| Dashboard | Streamlit |
| Data pipeline | Pandas, custom ETL scripts |

---

## Project Structure

```
WhaleBeaconNet/
├── notebooks/
│   ├── 01_eda_audio.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── preprocess_audio.py
│   ├── features.py
│   ├── model.py
│   ├── inference.py
│   └── ais_integration.py
├── data/
│   └── species_profiles/
├── app.py
├── requirements.txt
└── README.md
```

---

## Project Status

Work in progress. Core pipeline under construction.

- [x] Repository scaffold and README
- [x] Audio preprocessing pipeline (opus → wav → spectrogram)
- [ ] Dataset collection and labeling pipeline
- [ ] BirdNET fine-tuning
- [ ] Confidence thresholding and human review flag
- [ ] AIS integration layer
- [ ] Species acoustic knowledge base (for RAG)
- [ ] Streamlit dashboard
- [ ] End-to-end pipeline test

---

## Future Work

**Acoustic deterrent / whale-side warning system**

A natural extension would be emitting species-appropriate acoustic signals to guide whales away from high-risk corridors. This is intentionally deferred — each species responds differently to acoustic stimuli, and signals that deter one species may attract another (humpback whales in particular exhibit curiosity-driven approach behavior toward novel sounds). Continuous deterrents also cause chronic stress and habitat displacement. This direction requires collaboration with marine bioacoustics researchers before implementation.

---

## Related Projects

- [AnomalyAIS](https://github.com/vythwahh/AnomalyAIS) — AIS anomaly detection engine, used as the vessel-side integration layer
- [EcoCast](https://github.com/vythwahh/EcoCast) — Coral bleaching forecasting system with NOAA DHW pipeline and agentic Claude layer
- [CoralOpt](https://github.com/vythwahh/Coral-Reef-Conservation-Optimizer) — Coral reef conservation optimizer with real-time SST streaming

---

## Author

**Nguyen Trieu Vy Thu**
Mathematics & Computer Science — Data Science, HCMUS
Interested in the intersection of data engineering, machine learning, and marine conservation.
