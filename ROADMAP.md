# MangaRelief - Development Roadmap 🚀

Welcome to the MangaRelief roadmap! This document outlines the planned evolution of the project, taking it from a basic proof-of-concept to a professional 3D printing utility for manga panels and TCG accessories. 

The development is structured into incremental "Goals" (or Tiers), focusing first on performance, then on smart workflows, and finally on mass production and functional objects.

## 🥉 Goal 1: Core Foundation (Completed)
*The fundamental proof of concept.*
- [x] Load manga/comic images and extract grayscale data.
- [x] Manual assignment of color levels (Black, Midtones, White).
- [x] Generate a raw 3D mesh (STL) based on pixel brightness.
- [x] Basic UI structure.

## 🥈 Goal 2: Performance & Precision
*Making the software fast, lightweight, and print-ready.*
- [x] **Brutal Decimation Algorithm:** Implement edge-collapse algorithms (`fast_simplification`) to drastically reduce file sizes (e.g., from 200MB to 15MB) without losing visual details. Cap at 150k triangles for slicer stability.
- [x] **Smart Pixel Clipping:** Hard cutoffs for pure black (top layer) and pure white (base layer) to eliminate JPEG artifacts and create perfectly flat surfaces.
- [x] **Midtone Snapping:** Tolerance-based leveling for gray areas to avoid micro-stepping and ensure smooth color transitions.

## 🥇 Goal 3: Smart Workflow (Studio Automation)
*Reducing user friction with intelligent automation.*
- [x] **Auto-Detect Midtones (AI/Math):** Implement K-Means clustering (via OpenCV) to automatically analyze the image and set the optimal grayscale targets instantly.
- [x] **Dynamic UI:** Interface elements (like extra midtone sliders) automatically hide or appear based on the chosen complexity (2-color, 3-color, 4-color modes).
- [x] **Auto-White Suggester:** Histogram analysis to recommend the perfect threshold for cutting out background noise.

## 💎 Goal 4: Studio Edition
*Seamless integration with modern multicolor 3D printers.*
- [x] **.3MF Export:** Generate native `.3mf` files (instead of just STLs) to carry forward precise layer height information.
- [ ] **Slicer Profile Integration:** (Future iteration) Reverse-engineer Bambu Studio/PrusaSlicer XML configs to embed color changes directly into the .3mf file, requiring zero manual setup in the slicer.

## 🏭 Goal 5: Volume Producer (Batch Processing)
*Designed for Etsy sellers and power users.*
- [ ] **Drag & Drop Folder Support:** Load entire manga chapters at once.
- [ ] **Automated Queue:** Apply the K-Means automation and clipping rules to generate optimized 3MF files for dozens of pages sequentially in the background.

## 🐉 Goal 6: The Forge (Functional Objects)
*Moving beyond flat panels into customized 3D accessories.*
- [ ] **Boolean Engine Integration:** Combine the generated 3D manga relief with pre-existing functional STLs.
- [ ] **TCG Deckboxes:** Automatically fuse the generated relief onto the front face of a Trading Card Game deckbox (e.g., applying a Blue-Eyes White Dragon art to a Yu-Gi-Oh! deckbox).
- [ ] **Smartphone Covers & Bookmarks:** Templates to apply the reliefs to everyday printable objects.