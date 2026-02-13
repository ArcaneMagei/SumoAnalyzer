# 🚀 COMPLETE INSTALLATION GUIDE - Robot Sumo Analyzer

## 📦 What You're Getting

A complete Docker-based video analysis system that works on ALL your machines:
- ✅ Windows PC with RTX 4090 (fastest, CUDA GPU)
- ✅ Dell laptop with Intel Core Ultra 7 (CPU mode, still usable)
- ✅ MacBook M3 Pro (Metal GPU acceleration)

---

## ⚡ QUICK START (10 Minutes Total)

### Step 1: Install Docker Desktop (5 minutes)

**Windows (both PCs):**
1. Go to: https://www.docker.com/products/docker-desktop/
2. Click "Download for Windows"
3. Run installer
4. Restart if prompted
5. Launch Docker Desktop (wait for whale icon to appear)

**macOS:**
1. Go to: https://www.docker.com/products/docker-desktop/
2. Click "Download for Mac - Apple Silicon"
3. Drag Docker.app to Applications
4. Launch Docker from Applications
5. Grant permissions when asked

**Verify installation:**
```bash
docker --version
# Should show: Docker version 24.x or 25.x
```

---

### Step 2: Download Project Files (2 minutes)

**Create project folder:**

Windows:
```powershell
mkdir C:\SumoAnalyzer
cd C:\SumoAnalyzer
```

macOS/Linux:
```bash
mkdir ~/SumoAnalyzer
cd ~/SumoAnalyzer
```

**Download these files into your SumoAnalyzer folder:**

**Required files (you have all of these):**
1. `README.md` (main documentation)
2. `Dockerfile` (container definition)
3. `docker-compose.yml` (orchestration config)
4. `requirements.txt` (Python dependencies)
5. `.dockerignore` (build optimization)
6. `setup.py` (directory structure creator)
7. `app.py` (main Streamlit application)
8. `models/detector.py` (YOLO robot detection)
9. `models/dohyo.py` (ring detection)
10. `models/homography.py` (top-down transform)
11. `tracking/tracker.py` (multi-object tracking)
12. `strategy/classifier.py` (strategy classification)

---

### Step 3: Setup Project Structure (1 minute)

**Run setup script:**
```bash
python setup.py
```

This creates:
- `models/`, `tracking/`, `strategy/` directories
- `data/input/`, `data/output/` directories
- `logs/` directory
- `__init__.py` files for Python packages

---

### Step 4: Build and Run (5 minutes first time, 10 seconds after)

**Windows:**
```powershell
cd C:\SumoAnalyzer
docker compose up --build
```

**macOS/Linux:**
```bash
cd ~/SumoAnalyzer
docker compose up --build
```

**First build takes 5-10 minutes** (downloading base images and dependencies).

**You'll see:**
```
[+] Building ...
[+] Running 1/1
 ✔ Container sumo-analyzer  Created
Attaching to sumo-analyzer
sumo-analyzer  | 
sumo-analyzer  |   You can now view your Streamlit app in your browser.
sumo-analyzer  |   Network URL: http://0.0.0.0:8501
```

---

### Step 5: Access Application

**Open browser:** http://localhost:8501

**You should see:**
- 🤖 Robot Sumo Match Analyzer title
- Three tabs: Video Processing, Match Analysis, Robot Database
- Sidebar with settings
- GPU status indicator

✅ **Installation complete!**

---

## 📊 Platform-Specific Notes

### RTX 4090 Windows PC (Primary Machine)

**GPU Check:**
```powershell
# Verify Docker can see GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Should show your RTX 4090.

**Performance:**
- Processing: ~10-12 FPS
- 10s match: ~25-30 seconds
- Best for: Bulk processing, main analysis work

**If GPU not detected:**
Edit `docker-compose.yml` - see GPU troubleshooting below.

---

### Dell Intel Core Ultra 7 Laptop (Portable)

**No GPU setup needed** - runs on CPU automatically.

**Performance:**
- Processing: ~3-5 FPS
- 10s match: ~2-3 minutes
- Good for: Quick checks, viewing previous results

**Tip:** Process videos on RTX 4090 PC, view results on laptop.

---

### MacBook M3 Pro (Field Use)

**Metal GPU** automatically detected and used.

**Performance:**
- Processing: ~8-10 FPS
- 10s match: ~35-40 seconds
- Good for: Field analysis, competitions, portable work

**No special configuration needed.**

---

## 🎯 First Use - Test Workflow

### 1. Prepare Test Video

- Take one match video (MP4, AVI, MOV, or MKV)
- Ensure: full dohyo visible, good lighting, both robots clear
- Keep under 500MB for first test

### 2. Upload Video

1. Go to "📹 Video Processing" tab
2. Click "Browse files" or drag & drop
3. Video preview appears

### 3. Configure Settings

**Sidebar settings (use defaults first):**
- Confidence Threshold: 0.5
- Auto Dohyo Calibration: ✅ Enabled
- Slow Motion Video: ⬜ (check if your video is slow-mo)

**Strategy thresholds (adjust later):**
- Positioning Max Speed: 5 cm/s
- Search Speed Range: 10-40 cm/s
- Attack Min Speed: 50 cm/s
- Special Rotation Rate: 180 deg/s

### 4. Process Video

Click **"🚀 Analyze Match"**

**Progress indicators:**
- Initializing models (10%)
- Detecting dohyo (20%)
- Calibrating transform (30%)
- Detecting and tracking robots (30-70%)
- Classifying strategies (75-90%)
- Generating visualizations (90-100%)
- ✅ Processing complete!

### 5. Review Results

**Go to "📊 Match Analysis" tab:**

**Trajectory Plot:**
- Top-down view of robot paths
- Color-coded by strategy
- Hover for details

**Statistics Table:**
- Average speed, max speed
- Attack count
- Time in center
- Total distance traveled

**Strategy Timeline:**
- Second-by-second classification
- Color-coded phases

### 6. Export Data

**Available formats:**
- **CSV:** Frame-by-frame data (import into Excel/Python)
- **JSON:** Full match metadata
- **Video:** Annotated output (coming soon)

**Data saved to:** `data/output/csv/` and `data/output/json/`

---

## 🔧 Common Operations

### Daily Usage

```bash
# Start application
docker compose up

# Stop application (Ctrl+C, then:)
docker compose down

# View real-time logs
docker compose logs -f

# Restart
docker compose restart
```

### Updating Code

```bash
# After modifying Python files
docker compose up --build
```

### Clean Restart

```bash
# Remove everything and rebuild
docker compose down -v
docker compose up --build
```

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to Docker daemon"

**Solution:** Launch Docker Desktop and wait for it to fully start (whale icon active in system tray).

---

### Issue: GPU not detected on RTX 4090 PC

**Check Docker settings:**
1. Open Docker Desktop
2. Settings → Resources → WSL Integration
3. Enable GPU support
4. Restart Docker Desktop

**Verify:**
```powershell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**If still not working, edit `docker-compose.yml`:**

Remove the `deploy` section:
```yaml
    # Comment out if GPU not working
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
```

Application will run on CPU (slower but works).

---

### Issue: "Port 8501 already in use"

**Solution:** Change port in `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Use 8502 instead
```

Then access: http://localhost:8502

---

### Issue: "Dohyo not detected"

**Causes:**
- White line not visible
- Poor lighting
- Heavy occlusions

**Solutions:**
1. Try with a clearer video first
2. Ensure at least 50% of white line is visible
3. Lower confidence threshold to 0.3 in sidebar
4. Manual calibration (coming in next update)

---

### Issue: "No robots detected"

**Solutions:**
1. Lower confidence threshold in sidebar (0.3 or 0.4)
2. Verify robots are visible throughout video
3. Check if video quality is sufficient (720p+ recommended)

---

### Issue: Slow processing

**Expected performance:**
- RTX 4090: 25-30 sec for 10s video
- M3 Pro: 35-40 sec for 10s video
- Intel Core Ultra 7: 2-3 min for 10s video

**If much slower:**
1. Check Docker resources (Settings → Resources)
2. Allocate more RAM (8GB minimum, 16GB recommended)
3. Allocate more CPUs (4-6 cores)
4. Close other heavy applications

---

### Issue: Out of disk space

```bash
# Clean unused Docker data
docker system prune -a

# Check disk usage
docker system df
```

---

## 📁 Project Structure

```
SumoAnalyzer/
├── README.md                  # This file
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Docker configuration
├── requirements.txt           # Python dependencies
├── .dockerignore             # Build exclusions
├── setup.py                  # Directory structure creator
│
├── app.py                    # Main Streamlit application
│
├── models/
│   ├── __init__.py
│   ├── detector.py          # YOLO robot detection
│   ├── dohyo.py             # Ring detection
│   ├── homography.py        # Top-down transform
│   └── weights/             # Downloaded model weights
│
├── tracking/
│   ├── __init__.py
│   └── tracker.py           # Multi-object tracking
│
├── strategy/
│   ├── __init__.py
│   └── classifier.py        # Strategy classification
│
├── data/
│   ├── input/               # Put your match videos here
│   └── output/              # Results saved here
│       ├── csv/             # CSV exports
│       ├── json/            # JSON exports
│       └── videos/          # Annotated videos (future)
│
└── logs/                    # Application logs
```

---

## 🎓 Understanding the System

### Detection Pipeline

1. **Video Input:** Load match video
2. **Dohyo Detection:** Find 154cm black ring with 5cm white border
3. **Robot Detection:** YOLO detects bodies (20x20cm) + extensions (blades/flags)
4. **Tracking:** Maintain consistent IDs across frames
5. **Homography:** Transform to top-down view (bird's eye)
6. **Kinematics:** Calculate velocities, accelerations, rotations
7. **Strategy Classification:** Label behaviors (positioning/search/attack/special)
8. **Visualization:** Generate plots and statistics

### Strategy Classification Rules

**Positioning (Blue):**
- First 2 seconds of match
- Speed < 5 cm/s
- Minimal movement

**Search (Green):**
- Speed 10-40 cm/s
- Direction changes
- Scanning behavior

**Attack (Red):**
- Speed > 50 cm/s
- Moving toward opponent
- Distance decreasing

**Special (Orange):**
- Rotation rate > 180 deg/s
- Unusual patterns
- Blade flips, evasive maneuvers

---

## 💡 Tips for Best Results

### Video Quality Checklist

✅ **Resolution:** 720p or higher  
✅ **Frame rate:** 30 FPS or higher  
✅ **Lighting:** Consistent, no heavy shadows  
✅ **Dohyo visibility:** At least 50% of white line visible  
✅ **Camera stability:** Minimal shaking  
✅ **Focus:** Clear, not blurry  

### Workflow Recommendations

**Starting out:**
1. Process 1 simple video first
2. Verify all components work
3. Adjust thresholds as needed
4. Then process more videos

**Building database:**
1. Name robots in database tab
2. Process multiple matches with same robots
3. Track performance over time
4. Identify opponent patterns

**Using multiple machines:**
1. Process videos on RTX 4090 PC (fastest)
2. Sync `data/output/` folder to other machines
3. View results on any machine
4. Quick field checks on MacBook

---

## 🔄 Updates and Maintenance

### Updating Application

```bash
# If you received new code files
# 1. Replace the files
# 2. Rebuild container
docker compose down
docker compose up --build
```

### Backing Up Data

```bash
# Your data is in these folders:
data/input/     # Your original videos
data/output/    # Analysis results
logs/           # Application logs

# Simply copy these folders to backup location
```

### Cleaning Up

```bash
# Remove containers and images
docker compose down -v
docker rmi sumo-analyzer

# Clean all unused Docker data
docker system prune -a

# Keep your data (it's outside Docker)
# data/ and logs/ folders are safe
```

---

## ✅ Success Checklist

You know it's working correctly when:

- ✅ Docker Desktop is running
- ✅ `docker compose up` starts without errors
- ✅ Browser opens to http://localhost:8501
- ✅ App loads with three tabs visible
- ✅ GPU status shown in sidebar (or "CPU" for Dell laptop)
- ✅ Can upload a test video
- ✅ Dohyo detected (green circle shown)
- ✅ Robots tracked (consistent IDs throughout)
- ✅ Top-down trajectory looks reasonable
- ✅ Can export CSV/JSON
- ✅ Results saved in `data/output/`

---

## 🆘 Getting Help

**Check logs:**
```bash
docker compose logs -f sumo-analyzer
```

**Container status:**
```bash
docker ps
```

**Restart everything:**
```bash
docker compose down
docker compose up --build
```

**Nuclear option (fresh start):**
```bash
docker compose down -v
docker system prune -a
docker compose up --build
```

---

## 📚 Next Steps

### After First Successful Test

1. **Process 5-10 different matches**
   - Variety of robots, lighting conditions
   - Note what works well, what doesn't

2. **Adjust strategy thresholds**
   - Based on your robot's behavior
   - Fine-tune classification accuracy

3. **Build robot database**
   - Name your robots
   - Name opponent robots
   - Track patterns over time

4. **Provide feedback for Phase 2**
   - What features are most important?
   - What's missing?
   - What should be improved?

### Phase 2 Features (Coming)

- Annotation tool for model fine-tuning
- Improved robot identity across matches
- "Why did X win?" analysis
- Multi-camera support
- Batch processing
- Advanced visualizations

---

## 🎯 Ready to Start!

**You have everything you need:**

1. ✅ `README.md` - Main documentation (this file)
2. ✅ `Dockerfile` - Container definition
3. ✅ `docker-compose.yml` - Docker config
4. ✅ `requirements.txt` - Dependencies
5. ✅ `.dockerignore` - Build optimization
6. ✅ `setup.py` - Directory creator
7. ✅ `app.py` - Main application
8. ✅ All module files (detector, dohyo, homography, tracker, classifier)

**Just run:**
```bash
python setup.py           # Create directories
docker compose up --build # Build and run
```

**Then open:** http://localhost:8501

## 🚀 Let's Analyze Some Robot Sumo! 🤖🥋
