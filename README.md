# MX Anticheat (originally Mineland Xcalibur)

![Java](https://img.shields.io/badge/Java-1.8%2B-ed8b00?style=for-the-badge&logo=java&logoColor=white)
![Version](https://img.shields.io/badge/Version-5.0-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-Unlicense-green?style=for-the-badge)
![AI](https://img.shields.io/badge/AI-Millennium_5-purple?style=for-the-badge)

> **Advanced entropy analysis powered by local Recurrent Neural Networks (RNN)**.

MX is a free, open-source, **Zero-Dependency** Machine Learning anticheat for Minecraft (1.8 - 26.3). 
Powered by powerful heuristic algorithms combined with machine learning on a self-written RNN (Millennium 5) basis.

---

## 🧠 The Millennium 5 Engine

MX is built upon the **Millennium 5** self-written open-source library. It is a plug&play fully custom Deep Learning framework written in Java for CPU.

### Key Technologies:
*   **Local Processing:** No external Python scripts, no GPUs, no cloud API calls required. Everything runs asynchronously on your server's CPU.
*   **Architecture:** Stacked Bi-LSTM (Bidirectional) with Attention Mechanism, Layer Normalization, AdamW.
*   **Model:** `quark-e-1.0-56k-public` (56,000 parameters) — capable of detecting many 'clever' hacks.
*   **Advanced Heuristic:** From basic to advanced heuristics, perfectly complemented by machine learning.

---

## 🚀 Getting Started

### Installation
1. Download the latest release `.jar`.
2. Drop it into your `plugins` folder.
3. Restart the server.
4. **Done.** The pre-trained weights are loaded automatically from JAR.

### Training Your Own AI (Optional)
MX allows you to train the neural network directly on your server to adapt to new cheats.

1.  **Collect Data:**
    *   Record a cheater: `/mx dataset cheat <player>`
    *   Record a legit player: `/mx dataset legit <player>`
    *   *Play for ~30-60 minutes to gather samples.*
2.  **Train:**
    *   Run `/mx train <model_index> <epochs>` (e.g., `/mx train 7 16`).
    *   The plugin will run the **AdamW** optimizer (Backpropagation) in the background.
3.  **Deploy:**
    *   The new weights are saved automatically.

---

## 💻 Commands

| Command | Permission | Description |
| :--- | :--- | :--- |
| `/mx` | `mx.admin` | Main help menu. |
| `/mx alerts` | `mx.admin` | Toggle violation alerts. |
| `/mx stats` | `mx.admin` | View global ban/flag statistics. |
| `/mx activity <player>` | `mx.admin` | Generate a visual graph of player rotations (Pastebin). |
| `/mx dataset <mode> <player>` | `mx.admin` | Start recording samples for ML training (`legit`/`cheat`/`off`). |
| `/mx train <index> <epochs>` | `mx.admin` | Start async training of the neural network. |
| `/mx ml <index> <param> <val>` | `mx.admin` | Tweak ML hyperparameters (Learning Rate, Dropout, etc.) live. |

---

## 📂 Configuration

MX is highly configurable.
*   **`checks.yml`**: Enable/Disable specific checks, tune statistical thresholds.
*   **`config.yml`**:
    *   `prevention`: Set the harshness of lag-backs (0 = Silent, 3 = Aggressive).
    *   `ignoreCinematic`: Reduce false positives for cinematic camera users.
    *   `rotationsContainer`: Enable storing movement history for the `/mx activity` command.

---

## 🤝 Contributing

MX is open-source under the **Unlicense**. You are free to fork, modify, sell, or do whatever you want with the code.

**We need your help:**
*   Submit datasets of new private cheats.
*   Share trained `.dat` model weights.
*   Improve the math logic.

---

<div align="center">

**Created by Kireiko Oleksandr (pawsashatoy)**

</div>
