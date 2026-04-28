# MangaRelief Pro - 2D to 3D Manga Panel Converter

🌍 **[Scroll down for English Version](#english-version)**

## 🇮🇹 Versione Italiana

### Introduzione
Benvenuto in **MangaRelief Pro**! Questo software stand-alone ti permette di trasformare una qualsiasi tavola manga o illustrazione in scala di grigi (2D) in un modello 3D in rilievo altamente compatto ed ottimizzato, pronto per la stampa 3D (file STL). Attraverso un algoritmo proprietario di quantizzazione a "terrazze", converte l'intensità dell'inchiostro nativo in spessore fisico, preservando i contorni senza spalmare i dettagli.

### Guida all'Uso (Step-by-Step)

#### 1. Caricamento dell'Immagine
Avvia il programma e premi su **📂 Carica Manga**. Il sistema supporta formati di base e formati Web di nuova generazione (.JPG, .PNG, .AVIF, .WEBP, .HEIC). L'immagine verrà importata senza perdere un singolo pixel, per darti un'anteprima assoluta su cui lavorare.

#### 2. Navigazione del Canvas
La finestra sinistra di anteprima integra strumenti visivi in perfetto stile CAD:
*   **Zoom Ottico**: Usa la **Rotella del Mouse** per ingrandire o rimpicciolire liberamente l'immagine.
*   **Pan (Spostamento)**: Mantieni premuto il **Tasto Destro** (o Centrale) del mouse e trascinalo per esplorare i dettagli nascosti quando sei ravvicinato.

#### 3. Sistema di Campionamento Colori (Color-Picking)
Ogni scannerizzazione o opera digitale ha un proprio livello di luminosità e contrasto. La barra laterale destra a 4 Swatch permette di assegnare *da quale livello di grigio* far scattare un gradino in rilievo sul modello 3D:
1.  Clicca su uno dei bottoni Swatch (es. **L0 (Bianco)**). Verrà bordato di colore.
2.  Passa con il mouse sull'immagine e **fai Clic con il Tasto Sinistro** sul pixel interessato (es. lo sfondo puro del foglio).
3.  Ripeti il procedimento catturando in ordine: `Sfondo`, `Ombre Chiare/Retini/Fumo`, `Ombre Medie` e infine `Nero/Inchiostrazione Pura`. I bottoni immagazzineranno automaticamente in memoria il codice sorgente del grigio catturato.

#### 4. Generazione File STL
Assicurati di impostare la grandezza massima in mm a cui vuoi orientare la stampa `Dim Max (mm)` (di default 20 cm, 200 mm) e lo spessore totale `Z Max (mm)` e di `Base`. 
Premi su **🚀 Genera STL**. 
*Nota: Anche su scanzioni 4K la memoria grafica non friggerà mai durante il rendering, poiché la barra di avanzamento (QThread asincrono) comprimerà opportunamente le maglie per la generazione preservando intatta l'app della UI. Il file verrà salvato in automatico in una cartella `output/` posta nello stesso percorso dell'immagine originale, e riceverà un numero progressivo anti-sovrascrittura se iteri i tuoi file senza spostarli.*

### Consigli per la Stampa 3D (Best Practices)
La geometria generata da *MangaRelief Pro* è una mesh *watertight* di precisione a 4 livelli netti, intesa per emulare la tecnica del Lihtophane frontale o dell'HueForge ad incisione rapida. Usa sempre questi setup nello *Slicer* (es. Bambu Studio, OrcaSlicer, PrusaSlicer, Cura):

*   **Altezza del Layer (Z)**: Per godere delle sfumature in scala di grigi, si consiglia un'altezza strato microscopica variabile da **0.08 mm** a **0.12 mm**. L'ugello (Nozzle) standard da 0.4mm va benissimo.
*   **Tecnica del Cambio Colore Parallelo (M600)**: Consigliamo l'uso del cambio filamento manuale o tramite moduli automatici basato sui layer.
    *   *Suggerimento base:* Stampa lo strato di `Base` in Filamento Bianco (0mm -> 1.0mm). Inserisci una pausa al layer (M600) e cambia in Filamento Nero per i livelli in rilievo (1.0mm -> 2.5mm).
*   **Generatore di Pareti (Wall Generator)**: Abilita il motore ibrido **Arachne**. Arachne adegua la larghezza della cordolo di estrusione dinamicamente gestendo nativamente tutti i micropunti di inchiostrazione del pennino senza lasciare vuoti da "fill".

### Note Tecniche & Supporto
**Sistema Operativo Target**: Windows 10/11 (Compilazione nativa).
In caso di bugs, si prega di estrapolare i log di PyQT e inviarli su X all'autore di riferimento, che non vediamo l'ora di conoscerti!

---

## 🇬🇧 English Version

### Introduction
Welcome to **MangaRelief Pro**! This stand-alone software turns any manga panel or grayscale artwork (2D) into a highly robust, 3D printable relief model (STL file). Using a proprietary "terracing" layer-quantisation backend algorithm, it accurately interpolates native ink contrasts into physical layers, avoiding smoothed or smeared topographical features.

### Step-by-Step User Guide

#### 1. Importing the Image
Launch the application and click **📂 Carica Manga** (Load Manga). Base setups and newer generation formats (.JPG, .PNG, .AVIF, .WEBP, .HEIC) are fully supported. 

#### 2. Canvas Interaction
The left viewport allows for completely decoupled CAD visual tools without destroying resolution:
*   **Optical Zoom**: Keep scrolling the **Mouse Wheel** to punch into ink details instantly.
*   **Drag Panning**: Hold the **Right Click** (or Middle Click) and drag around your mouse pointer to pan all over the close-up viewport.

#### 3. Color Picking Calibration workflow
No scan contrast is identical to another. Setting up strict brightness bins handles any lighting condition possible from manual paper scan configurations:
1.  Click on any Swatch button from the right-hand bar (e.g. **L0 (Bianco)**). An outline denotes the active listener.
2.  Aim the reticle to your 2D preview and **Left Click** roughly on the targeted gray tone zone for that layer. 
3.  Gather your points accordingly (Paper Background -> Fine screentones / Smokes -> Heavy Shadows -> Strong Inks).

#### 4. STL Generation Trigger
Amend `Dim Max (mm)` sizes (default 20cm/200mmm length keeping intact aspect ratios seamlessly) or the targeted 3D plate thickness using the numeric inputs. Apply it using **🚀 Genera STL**.
The file outputs natively right beside your original image layout inside an `output/` sub-folder, automatically appending numbers if duplicates are rendered. The underlying QThreads ensure that PC freezes and huge RAM spikes stay entirely mitigated whilst leaving you with a dynamic real-time Progress Bar indicating what the Trimesh engine is precisely up to.

### 3D Printing Best Practices
Generated meshes are completely manifold out of the box explicitly scaled to match multi-layer swapping prints. Please adhere to the following generic setups for the Slicers (Bambu Studio, PrusaSlicer, Orca, Cura):

*   **Layer Height (Z)**: Target **0.08 mm** to **0.12 mm**. An average 0.4 mm brass nozzle acts more than adequate without hardware replacement needed.
*   **Dual Color Swapping Tech (M600 layer pause)**: Layer changing grants outstanding multi-colour illusion prints resembling standard comic sheets heavily inked. Print your baseline (e.g. up to 1.0mm) using White Filament, insert a custom Pause order on the Slicer, swap your spool manually (or setup your hardware AMS/MMU) and resume with Black or Dark Grey filament for everything atop 1.0mm marking the highest cliffs (the 2.5mm dark ink marks on top).
*   **Wall Thickness Generators**: Make double sure you're operating with an **Arachne** engine toggle active in your Slicer Settings. It dramatically sharpens incredibly thin raster lines rendering sharp manga pen points natively without tiny gap fills.

### Tech Specs
**OS Targets**: Packaged for Windows 10/11. 
For assistance ping the author via their Gumroad distribution portal!
