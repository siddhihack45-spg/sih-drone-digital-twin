/* =========================================================
   AEROVISION
   Autonomous Drone Digital Twin
   ========================================================= */


/* ================= GLOBAL STATE ================= */

let scene;
let camera;
let renderer;
let controls;
let gridHelper;

let cloudGroup;
let trajectoryGroup;
let externalModelGroup;

let pointCloud = null;
let trajectoryLine = null;

let trajectoryData = [];

let gpsAvailable = false;
let imuAvailable = false;

let hasConfidenceField = false;

let currentViewMode = "pointcloud";

let toastTimer = null;


/* ================= INITIALIZATION ================= */

document.addEventListener("DOMContentLoaded", () => {

    initClock();

    initViewer();

    setupNavigation();

    setupFileInputs();

    loadReconstruction();

    loadTrajectory();

    detectSensors();

    logMessage("SYSTEM", "AEROVISION mission control initialized.");
    logMessage("VIEWER", "Connecting to reconstruction.ply...");
    logMessage("TRAJECTORY", "Connecting to fused_trajectory.csv...");

});


/* ================= CLOCK ================= */

function initClock() {

    function updateClock() {

        const now = new Date();

        const h = String(now.getUTCHours()).padStart(2, "0");
        const m = String(now.getUTCMinutes()).padStart(2, "0");
        const s = String(now.getUTCSeconds()).padStart(2, "0");

        const clock = document.getElementById("utcClock");

        if (clock) {
            clock.textContent = `${h}:${m}:${s} UTC`;
        }

    }

    updateClock();

    setInterval(updateClock, 1000);
}


/* ================= NAVIGATION ================= */

function setupNavigation() {

    const links = document.querySelectorAll(".nav-item");

    links.forEach(link => {

        link.addEventListener("click", () => {

            links.forEach(item => item.classList.remove("active"));

            link.classList.add("active");

        });

    });

}


function scrollToSection(id) {

    const element = document.getElementById(id);

    if (!element) return;

    element.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

}


/* ================= THREE.JS VIEWER ================= */

function initViewer() {

    const container = document.getElementById("three-container");

    if (!container) return;


    scene = new THREE.Scene();

    scene.background = new THREE.Color(0x070707);


    camera = new THREE.PerspectiveCamera(
        55,
        container.clientWidth / container.clientHeight,
        0.01,
        10000
    );

    camera.position.set(3.5, 3, 5);


    renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: false
    });

    renderer.setPixelRatio(
        Math.min(window.devicePixelRatio, 2)
    );

    renderer.setSize(
        container.clientWidth,
        container.clientHeight
    );

    renderer.outputEncoding = THREE.sRGBEncoding;

    container.appendChild(renderer.domElement);


    controls = new THREE.OrbitControls(
        camera,
        renderer.domElement
    );

    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    controls.minDistance = 0.1;
    controls.maxDistance = 10000;


    /* Groups */

    cloudGroup = new THREE.Group();
    trajectoryGroup = new THREE.Group();
    externalModelGroup = new THREE.Group();

    scene.add(cloudGroup);
    scene.add(trajectoryGroup);
    scene.add(externalModelGroup);


    /* Grid */

    gridHelper = new THREE.GridHelper(
        20,
        20,
        0x444444,
        0x222222
    );

    gridHelper.position.y = -0.05;

    scene.add(gridHelper);


    /* Lighting */

    const ambient = new THREE.AmbientLight(
        0xffffff,
        1
    );

    scene.add(ambient);


    window.addEventListener(
        "resize",
        onViewerResize
    );


    animateViewer();

}


function animateViewer() {

    requestAnimationFrame(animateViewer);

    if (controls) {
        controls.update();
    }

    if (renderer && scene && camera) {

        renderer.render(
            scene,
            camera
        );

    }

}


function onViewerResize() {

    const container = document.getElementById(
        "three-container"
    );

    if (!container || !camera || !renderer) return;

    camera.aspect =
        container.clientWidth /
        container.clientHeight;

    camera.updateProjectionMatrix();

    renderer.setSize(
        container.clientWidth,
        container.clientHeight
    );

}


/* ================= RECONSTRUCTION ================= */

function loadReconstruction() {

    const loader = new THREE.PLYLoader();

    const path = "output/reconstruction.ply";


    loader.load(

        path,

        geometry => {

            clearPointCloud();


            const position = geometry.getAttribute(
                "position"
            );

            if (!position) {

                showToast(
                    "PLY has no position attribute."
                );

                return;

            }


            const colors = geometry.getAttribute(
                "color"
            );

            hasConfidenceField =
                !!geometry.getAttribute("confidence");


            geometry.computeBoundingBox();

            geometry.computeBoundingSphere();

            geometry.center();


            const pointCount =
                position.count;


            const hasColor =
                !!colors;


            const radius =
                geometry.boundingSphere
                    ? geometry.boundingSphere.radius
                    : 1;


            const pointSize =
                Math.max(
                    radius / 180,
                    0.008
                );


            const material =
                new THREE.PointsMaterial({

                    size: pointSize,

                    sizeAttenuation: true,

                    color:
                        hasColor
                            ? 0xffffff
                            : 0xf0f0f0,

                    vertexColors: hasColor,

                    transparent: true,

                    opacity: 0.92

                });


            pointCloud =
                new THREE.Points(
                    geometry,
                    material
                );


            pointCloud.name =
                "ReconstructionPointCloud";


            pointCloud.userData.hasColor =
                hasColor;


            cloudGroup.add(
                pointCloud
            );


            updatePointCloudUI(
                pointCount,
                hasColor
            );


            rebuildTrajectoryLine();

            fitView();

            hideViewerLoading();

            document.getElementById(
                "viewerState"
            ).textContent = "RECONSTRUCTION LOADED";


            setModuleStatus(
                "status-recon",
                "READY / PLY LOADED"
            );


            logMessage(
                "RECON",
                `${formatNumber(pointCount)} 3D points loaded.`
            );


            showToast(
                "3D reconstruction loaded."
            );

        },

        xhr => {

            if (!xhr.total) return;

            const percent =
                Math.round(
                    (xhr.loaded / xhr.total) * 100
                );

            const state =
                document.getElementById(
                    "viewerState"
                );

            if (state) {
                state.textContent =
                    `LOADING ${percent}%`;
            }

        },

        error => {

            console.error(
                "PLY loading error:",
                error
            );

            hideViewerLoading();

            document.getElementById(
                "viewerState"
            ).textContent =
                "PLY NOT FOUND";


            setModuleStatus(
                "status-recon",
                "OUTPUT NOT FOUND"
            );


            logMessage(
                "RECON",
                "Could not load output/reconstruction.ply."
            );


            showToast(
                "reconstruction.ply could not be loaded."
            );

        }

    );

}


/* ================= POINT CLOUD ================= */

function clearPointCloud() {

    if (!cloudGroup) return;

    while (cloudGroup.children.length) {

        const object =
            cloudGroup.children[0];

        cloudGroup.remove(object);

        if (object.geometry) {
            object.geometry.dispose();
        }

        if (object.material) {

            if (Array.isArray(object.material)) {

                object.material.forEach(
                    material => material.dispose()
                );

            } else {

                object.material.dispose();

            }

        }

    }

    pointCloud = null;

}


function updatePointCloudUI(
    pointCount,
    hasColor
) {

    const formatted =
        formatNumber(pointCount);


    setText(
        "pointCount",
        formatted
    );

    setText(
        "hudPoints",
        formatted
    );


    setText(
        "geometryMetric",
        "AVAILABLE"
    );


    setText(
        "rgbMetric",
        hasColor
            ? "AVAILABLE"
            : "NOT PRESENT"
    );


    setText(
        "colorState",
        hasColor
            ? "RGB"
            : "MONO"
    );


    setText(
        "confidenceMetric",
        hasConfidenceField
            ? "AVAILABLE"
            : "NOT ATTACHED"
    );


    setText(
        "geometryType",
        "POINT CLOUD"
    );


    updateModuleStatus(
        "status-confidence",
        hasConfidenceField
            ? "CONFIDENCE FIELD"
            : "QA READY"
    );

}


/* ================= TRAJECTORY ================= */

async function loadTrajectory() {

    try {

        const response =
            await fetch(
                "output/fused_trajectory.csv"
            );


        if (!response.ok) {
            throw new Error(
                "Trajectory CSV unavailable"
            );
        }


        const text =
            await response.text();


        parseTrajectoryCSV(text);


        logMessage(
            "TRAJECTORY",
            `${trajectoryData.length} pose samples loaded.`
        );


        setModuleStatus(
            "status-slam",
            "VO TRAJECTORY LOADED"
        );


    } catch (error) {

        console.warn(error);

        logMessage(
            "TRAJECTORY",
            "fused_trajectory.csv not available."
        );


        setModuleStatus(
            "status-slam",
            "WAITING FOR TRAJECTORY"
        );

    }

}


function parseTrajectoryCSV(text) {

    const lines =
        text
            .trim()
            .split(/\r?\n/);


    trajectoryData = [];


    if (lines.length < 2) return;


    for (let i = 1; i < lines.length; i++) {

        const row =
            lines[i]
                .split(",")
                .map(value => value.trim());


        if (row.length < 4) continue;


        const frame =
            Number(row[0]);

        const x =
            Number(row[1]);

        const y =
            Number(row[2]);

        const z =
            Number(row[3]);


        if (
            Number.isFinite(frame) &&
            Number.isFinite(x) &&
            Number.isFinite(y) &&
            Number.isFinite(z)
        ) {

            trajectoryData.push({
                frame,
                x,
                y,
                z
            });

        }

    }


    const count =
        trajectoryData.length;


    setText(
        "trajectoryCount",
        formatNumber(count)
    );

    setText(
        "trajectoryBig",
        formatNumber(count)
    );

    setText(
        "visionTrajectory",
        `${formatNumber(count)} POSES`
    );


    setText(
        "sideFrameCount",
        formatNumber(count)
    );


    setText(
        "frameCount",
        formatNumber(count)
    );


    setText(
        "frameCount",
        formatNumber(count)
    );


    updateCoverage();

    drawTrajectoryChart();

    rebuildTrajectoryLine();

}


function rebuildTrajectoryLine() {

    if (!trajectoryGroup) return;


    while (trajectoryGroup.children.length) {

        const object =
            trajectoryGroup.children[0];

        trajectoryGroup.remove(object);

        if (object.geometry) {
            object.geometry.dispose();
        }

        if (object.material) {
            object.material.dispose();
        }

    }

    trajectoryLine = null;


    if (trajectoryData.length < 2) {
        return;
    }


    /*
        The current project uses monocular visual
        odometry. Its scale is relative.

        We normalize the trajectory for a visual
        dashboard overlay.
    */

    const bounds =
        getTrajectoryBounds();


    const span =
        Math.max(
            bounds.maxX - bounds.minX,
            bounds.maxY - bounds.minY,
            bounds.maxZ - bounds.minZ,
            0.0001
        );


    let targetSize = 4;


    if (
        pointCloud &&
        pointCloud.geometry &&
        pointCloud.geometry.boundingSphere
    ) {

        targetSize =
            Math.max(
                pointCloud.geometry.boundingSphere.radius
                * 1.2,
                2
            );

    }


    const points = trajectoryData.map(
        item => {

            return new THREE.Vector3(

                (
                    item.x -
                    bounds.centerX
                ) /
                span *
                targetSize,

                (
                    item.y -
                    bounds.centerY
                ) /
                span *
                targetSize,

                (
                    item.z -
                    bounds.centerZ
                ) /
                span *
                targetSize

            );

        }
    );


    const geometry =
        new THREE.BufferGeometry()
            .setFromPoints(points);


    const material =
        new THREE.LineBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.85
        });


    trajectoryLine =
        new THREE.Line(
            geometry,
            material
        );


    trajectoryLine.name =
        "VisualTrajectory";


    trajectoryGroup.add(
        trajectoryLine
    );


    if (currentViewMode === "pointcloud") {

        trajectoryGroup.visible = false;

    } else {

        trajectoryGroup.visible = true;

    }

}


/* ================= TRAJECTORY HELPERS ================= */

function getTrajectoryBounds() {

    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;

    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;


    trajectoryData.forEach(point => {

        minX = Math.min(
            minX,
            point.x
        );

        minY = Math.min(
            minY,
            point.y
        );

        minZ = Math.min(
            minZ,
            point.z
        );

        maxX = Math.max(
            maxX,
            point.x
        );

        maxY = Math.max(
            maxY,
            point.y
        );

        maxZ = Math.max(
            maxZ,
            point.z
        );

    });


    return {

        minX,
        minY,
        minZ,

        maxX,
        maxY,
        maxZ,

        centerX:
            (minX + maxX) / 2,

        centerY:
            (minY + maxY) / 2,

        centerZ:
            (minZ + maxZ) / 2

    };

}


/* ================= 2D TRAJECTORY ================= */

function drawTrajectoryChart() {

    const canvas =
        document.getElementById(
            "trajectoryCanvas"
        );


    if (!canvas) return;


    const rect =
        canvas.getBoundingClientRect();


    const dpr =
        Math.min(
            window.devicePixelRatio || 1,
            2
        );


    canvas.width =
        rect.width * dpr;

    canvas.height =
        rect.height * dpr;


    const ctx =
        canvas.getContext("2d");


    ctx.scale(
        dpr,
        dpr
    );


    const width =
        rect.width;

    const height =
        rect.height;


    ctx.clearRect(
        0,
        0,
        width,
        height
    );


    /* Grid */

    ctx.strokeStyle =
        "rgba(255,255,255,0.06)";

    ctx.lineWidth = 1;


    const gridSize = 45;


    for (
        let x = 0;
        x <= width;
        x += gridSize
    ) {

        ctx.beginPath();

        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);

        ctx.stroke();

    }


    for (
        let y = 0;
        y <= height;
        y += gridSize
    ) {

        ctx.beginPath();

        ctx.moveTo(0, y);
        ctx.lineTo(width, y);

        ctx.stroke();

    }


    if (trajectoryData.length < 2) {

        ctx.fillStyle =
            "#555";

        ctx.font =
            "10px monospace";

        ctx.fillText(
            "WAITING FOR TRAJECTORY DATA",
            20,
            30
        );

        return;

    }


    const bounds =
        getTrajectoryBounds();


    const rangeX =
        Math.max(
            bounds.maxX - bounds.minX,
            0.0001
        );

    const rangeZ =
        Math.max(
            bounds.maxZ - bounds.minZ,
            0.0001
        );


    /*
        Prefer X/Z top-down projection.
        If Z is almost flat, use X/Y.
    */

    const useY =
        rangeZ <
        rangeX * 0.05;


    const secondMin =
        useY
            ? bounds.minY
            : bounds.minZ;

    const secondMax =
        useY
            ? bounds.maxY
            : bounds.maxZ;


    const rangeSecond =
        Math.max(
            secondMax - secondMin,
            0.0001
        );


    const padding = 35;


    const scale =
        Math.min(
            (width - padding * 2) /
                rangeX,

            (height - padding * 2) /
                rangeSecond
        );


    const path = [];


    trajectoryData.forEach(
        point => {

            const px =
                padding +
                (point.x - bounds.minX)
                * scale;


            const second =
                useY
                    ? point.y
                    : point.z;


            const py =
                height -
                padding -
                (second - secondMin)
                * scale;


            path.push({
                x: px,
                y: py
            });

        }
    );


    /* Path */

    ctx.beginPath();

    path.forEach(
        (point, index) => {

            if (index === 0) {

                ctx.moveTo(
                    point.x,
                    point.y
                );

            } else {

                ctx.lineTo(
                    point.x,
                    point.y
                );

            }

        }
    );


    ctx.strokeStyle =
        "#ffffff";

    ctx.lineWidth = 2;

    ctx.stroke();


    /* Start */

    const start =
        path[0];


    ctx.beginPath();

    ctx.arc(
        start.x,
        start.y,
        4,
        0,
        Math.PI * 2
    );

    ctx.fillStyle =
        "#d9ff68";

    ctx.fill();


    /* End */

    const end =
        path[path.length - 1];


    ctx.beginPath();

    ctx.arc(
        end.x,
        end.y,
        5,
        0,
        Math.PI * 2
    );

    ctx.fillStyle =
        "#ffffff";

    ctx.fill();


    /* Labels */

    ctx.font =
        "8px monospace";

    ctx.fillStyle =
        "#555";

    ctx.fillText(
        "START",
        start.x + 8,
        start.y - 8
    );

    ctx.fillText(
        "CURRENT",
        end.x + 8,
        end.y - 8
    );

}


/* ================= COVERAGE ================= */

function updateCoverage() {

    const frames =
        trajectoryData.length;


    if (!frames) return;


    /*
        Current project trajectory has one sample
        per extracted frame.
    */

    const coverage = 100;


    setText(
        "coverageValue",
        `${coverage}%`
    );

    setText(
        "analyticsCoverage",
        `${coverage}%`
    );


    const bar =
        document.getElementById(
            "coverageBar"
        );

    if (bar) {
        bar.style.width =
            `${coverage}%`;
    }

}


/* ================= SENSOR DETECTION ================= */

async function detectSensors() {

    await detectGPS();

    await detectIMU();

    updateFusionState();

}


async function detectGPS() {

    try {

        const response =
            await fetch(
                "../data/gps.csv",
                {
                    cache: "no-store"
                }
            );


        if (!response.ok) {
            throw new Error("GPS unavailable");
        }


        const text =
            await response.text();


        gpsAvailable = true;


        const rows =
            parseCSVRows(text);


        setText(
            "gpsStatus",
            "ON"
        );

        setText(
            "gpsPanelState",
            "DATA DETECTED"
        );

        setText(
            "gpsPanelDetail",
            `${rows.length} ROWS`
        );


        document.getElementById(
            "gpsBar"
        ).style.width = "100%";


        setText(
            "gpsPercent",
            "100%"
        );


        updateGPSPreview(
            rows
        );


        logMessage(
            "GPS",
            `GPS dataset detected (${rows.length} rows).`
        );


        setModuleStatus(
            "status-geo",
            "GPS DATA DETECTED"
        );


    } catch (error) {

        gpsAvailable = false;


        setText(
            "gpsStatus",
            "OFF"
        );

        setText(
            "gpsPanelState",
            "NOT DETECTED"
        );

        setText(
            "gpsPanelDetail",
            "WAITING"
        );


        document.getElementById(
            "gpsBar"
        ).style.width = "0%";


        setText(
            "gpsPercent",
            "0%"
        );

    }

}


async function detectIMU() {

    try {

        const response =
            await fetch(
                "../data/imu.csv",
                {
                    cache: "no-store"
                }
            );


        if (!response.ok) {
            throw new Error("IMU unavailable");
        }


        const text =
            await response.text();


        imuAvailable = true;


        const rows =
            parseCSVRows(text);


        setText(
            "imuStatus",
            "ON"
        );

        setText(
            "imuPanelState",
            "DATA DETECTED"
        );

        setText(
            "imuPanelDetail",
            `${rows.length} ROWS`
        );


        document.getElementById(
            "imuBar"
        ).style.width = "100%";


        setText(
            "imuPercent",
            "100%"
        );


        logMessage(
            "IMU",
            `IMU dataset detected (${rows.length} rows).`
        );


    } catch (error) {

        imuAvailable = false;


        setText(
            "imuStatus",
            "OFF"
        );

        setText(
            "imuPanelState",
            "NOT DETECTED"
        );

        setText(
            "imuPanelDetail",
            "WAITING"
        );


        document.getElementById(
            "imuBar"
        ).style.width = "0%";


        setText(
            "imuPercent",
            "0%"
        );

    }

}


function updateFusionState() {

    if (
        gpsAvailable &&
        imuAvailable
    ) {

        setText(
            "fusionPanelState",
            "SENSORS DETECTED"
        );

        setModuleStatus(
            "status-fusion",
            "GPS + IMU AVAILABLE"
        );

    } else if (
        gpsAvailable ||
        imuAvailable
    ) {

        setText(
            "fusionPanelState",
            "PARTIAL SENSOR INPUT"
        );

        setModuleStatus(
            "status-fusion",
            "PARTIAL SENSOR INPUT"
        );

    } else {

        setText(
            "fusionPanelState",
            "VISUAL FALLBACK"
        );

        setModuleStatus(
            "status-fusion",
            "VISUAL FALLBACK"
        );

    }

}


/* ================= GPS PREVIEW ================= */

function updateGPSPreview(rows) {

    if (!rows.length) return;


    /*
        Expected:
        frame,latitude,longitude,altitude
    */

    const header =
        Object.keys(rows[0]);


    const latKey =
        header.find(
            key =>
                key.toLowerCase()
                    .includes("latitude")
        );

    const lonKey =
        header.find(
            key =>
                key.toLowerCase()
                    .includes("longitude")
        );

    const altKey =
        header.find(
            key =>
                key.toLowerCase()
                    .includes("altitude")
        );


    const last =
        rows[rows.length - 1];


    if (latKey) {

        setText(
            "latitudeValue",
            Number(last[latKey])
                .toFixed(6)
        );

    }

    if (lonKey) {

        setText(
            "longitudeValue",
            Number(last[lonKey])
                .toFixed(6)
        );

    }

    if (altKey) {

        setText(
            "altitudeValue",
            Number(last[altKey])
                .toFixed(2)
        );

    }


    setText(
        "referenceValue",
        "GPS DATA"
    );


    setText(
        "geoDescription",
        "GPS dataset detected. Backend georeferencing can use these coordinates."
    );

}


/* ================= CSV ================= */

function parseCSVRows(text) {

    const lines =
        text
            .trim()
            .split(/\r?\n/);


    if (lines.length < 2) {
        return [];
    }


    const headers =
        lines[0]
            .split(",")
            .map(
                header =>
                    header.trim()
            );


    const rows = [];


    for (
        let i = 1;
        i < lines.length;
        i++
    ) {

        const values =
            lines[i]
                .split(",")
                .map(
                    value =>
                        value.trim()
                );


        if (
            values.length !==
            headers.length
        ) {
            continue;
        }


        const row = {};


        headers.forEach(
            (header, index) => {

                row[header] =
                    values[index];

            }
        );


        rows.push(row);

    }


    return rows;

}


/* ================= VIEW MODES ================= */

function setViewMode(mode) {

    currentViewMode =
        mode;


    document
        .querySelectorAll(".tool-btn")
        .forEach(
            button =>
                button.classList.remove("active")
        );


    if (mode === "pointcloud") {

        document
            .getElementById("viewPointCloud")
            .classList.add("active");

        cloudGroup.visible = true;
        trajectoryGroup.visible = false;
        externalModelGroup.visible = false;

        setText(
            "hudTrack",
            "POINT CLOUD"
        );

    }


    if (mode === "trajectory") {

        document
            .getElementById("viewTrajectory")
            .classList.add("active");

        cloudGroup.visible = false;
        trajectoryGroup.visible = true;
        externalModelGroup.visible = false;

        setText(
            "hudTrack",
            "TRAJECTORY"
        );

    }


    if (mode === "digital") {

        document
            .getElementById("viewDigital")
            .classList.add("active");

        cloudGroup.visible = true;
        trajectoryGroup.visible = true;
        externalModelGroup.visible = true;

        setText(
            "hudTrack",
            "DIGITAL TWIN"
        );

    }


    fitView();

}


function toggleGrid() {

    if (!gridHelper) return;

    gridHelper.visible =
        !gridHelper.visible;

}


function resetView() {

    camera.position.set(
        3.5,
        3,
        5
    );

    controls.target.set(
        0,
        0,
        0
    );

    controls.update();

}


function fitView() {

    if (
        !camera ||
        !controls ||
        !scene
    ) {
        return;
    }


    const box =
        new THREE.Box3();


    let hasObjects = false;


    if (
        cloudGroup &&
        cloudGroup.visible
    ) {

        box.expandByObject(
            cloudGroup
        );

        hasObjects = true;

    }


    if (
        trajectoryGroup &&
        trajectoryGroup.visible
    ) {

        box.expandByObject(
            trajectoryGroup
        );

        hasObjects = true;

    }


    if (
        externalModelGroup &&
        externalModelGroup.visible
    ) {

        box.expandByObject(
            externalModelGroup
        );

        hasObjects = true;

    }


    if (!hasObjects) return;


    const center =
        new THREE.Vector3();


    const size =
        new THREE.Vector3();


    box.getCenter(center);

    box.getSize(size);


    const maxSize =
        Math.max(
            size.x,
            size.y,
            size.z,
            0.5
        );


    const distance =
        maxSize * 1.8;


    camera.position.set(
        center.x + distance,
        center.y + distance * 0.7,
        center.z + distance
    );


    camera.near =
        Math.max(
            distance / 1000,
            0.001
        );

    camera.far =
        Math.max(
            distance * 100,
            1000
        );

    camera.updateProjectionMatrix();


    controls.target.copy(
        center
    );

    controls.update();

}


/* ================= DYNAMIC / CONFIDENCE ================= */

function toggleDynamicMode() {

    const enabled =
        document.getElementById(
            "dynamicToggle"
        ).checked;


    /*
        No dynamic-object mask is currently
        exported by the backend.

        This control communicates the pipeline
        state without falsely modifying geometry.
    */

    if (enabled) {

        setModuleStatus(
            "status-dynamic",
            "FILTER REQUESTED"
        );

        logMessage(
            "FILTER",
            "Dynamic-object filtering requested; mask output not attached."
        );

        showToast(
            "Dynamic mask awaiting backend output."
        );

    } else {

        setModuleStatus(
            "status-dynamic",
            "MASK READY"
        );

    }

}


function toggleConfidenceMode() {

    const enabled =
        document.getElementById(
            "confidenceToggle"
        ).checked;


    if (!pointCloud) return;


    if (enabled) {

        pointCloud.material.opacity =
            0.55;


        if (hasConfidenceField) {

            setModuleStatus(
                "status-confidence",
                "CONFIDENCE ACTIVE"
            );

        } else {

            setModuleStatus(
                "status-confidence",
                "VISUAL QA MODE"
            );

            showToast(
                "No confidence attribute found; visual QA mode enabled."
            );

        }

    } else {

        pointCloud.material.opacity =
            0.92;


        setModuleStatus(
            "status-confidence",
            hasConfidenceField
                ? "CONFIDENCE FIELD"
                : "QA READY"
        );

    }

}


/* ================= LOCAL FILES ================= */

function setupFileInputs() {

    const modelInput =
        document.getElementById(
            "modelFileInput"
        );

    const modelInputAlt =
        document.getElementById(
            "modelInputAlt"
        );


    if (modelInput) {

        modelInput.addEventListener(
            "change",
            event => {

                const file =
                    event.target.files[0];

                if (!file) return;

                loadLocalModel(file);

            }
        );

    }


    if (modelInputAlt) {

        modelInputAlt.addEventListener(
            "change",
            event => {

                const file =
                    event.target.files[0];

                if (!file) return;

                loadLocalModel(file);

            }
        );

    }


    const videoInput =
        document.getElementById(
            "videoInput"
        );


    if (videoInput) {

        videoInput.addEventListener(
            "change",
            event => {

                const file =
                    event.target.files[0];

                if (!file) return;

                logMessage(
                    "VIDEO",
                    `Local video selected: ${file.name}`
                );

                showToast(
                    `Selected ${file.name}`
                );

            }
        );

    }


    const gpsInput =
        document.getElementById(
            "gpsInput"
        );


    if (gpsInput) {

        gpsInput.addEventListener(
            "change",
            async event => {

                const file =
                    event.target.files[0];

                if (!file) return;

                const text =
                    await file.text();

                const rows =
                    parseCSVRows(text);


                gpsAvailable = true;


                setText(
                    "gpsStatus",
                    "ON"
                );

                setText(
                    "gpsPanelState",
                    "LOCAL FILE"
                );

                setText(
                    "gpsPanelDetail",
                    `${rows.length} ROWS`
                );


                updateGPSPreview(rows);

                updateFusionState();


                logMessage(
                    "GPS",
                    `Local GPS file loaded: ${file.name}`
                );


                showToast(
                    "GPS data loaded locally."
                );

            }
        );

    }


    const imuInput =
        document.getElementById(
            "imuInput"
        );


    if (imuInput) {

        imuInput.addEventListener(
            "change",
            async event => {

                const file =
                    event.target.files[0];

                if (!file) return;

                const text =
                    await file.text();

                const rows =
                    parseCSVRows(text);


                imuAvailable = true;


                setText(
                    "imuStatus",
                    "ON"
                );

                setText(
                    "imuPanelState",
                    "LOCAL FILE"
                );

                setText(
                    "imuPanelDetail",
                    `${rows.length} ROWS`
                );


                updateFusionState();


                logMessage(
                    "IMU",
                    `Local IMU file loaded: ${file.name}`
                );


                showToast(
                    "IMU data loaded locally."
                );

            }
        );

    }

}


/* ================= MODEL LOADER ================= */

function loadLocalModel(file) {

    const name =
        file.name.toLowerCase();


    const extension =
        name.split(".").pop();


    document.getElementById(
        "selectedModel"
    ).textContent =
        file.name;


    clearExternalModel();


    const objectURL =
        URL.createObjectURL(file);


    if (extension === "ply") {

        const loader =
            new THREE.PLYLoader();


        loader.load(
            objectURL,
            geometry => {

                geometry.computeBoundingSphere();

                geometry.center();


                const colors =
                    geometry.getAttribute(
                        "color"
                    );


                const material =
                    new THREE.PointsMaterial({

                        size:
                            Math.max(
                                (
                                    geometry
                                        .boundingSphere
                                        ?.radius || 1
                                ) / 180,
                                0.008
                            ),

                        vertexColors:
                            !!colors,

                        color:
                            colors
                                ? 0xffffff
                                : 0xf0f0f0,

                        sizeAttenuation: true

                    });


                const points =
                    new THREE.Points(
                        geometry,
                        material
                    );


                externalModelGroup.add(
                    points
                );


                document.getElementById(
                    "hudModel"
                ).textContent =
                    file.name.toUpperCase();


                setViewMode(
                    "digital"
                );


                fitView();


                URL.revokeObjectURL(
                    objectURL
                );


                logMessage(
                    "MODEL",
                    `${file.name} loaded.`
                );


                showToast(
                    "3D model loaded."
                );

            },

            undefined,

            error => {

                console.error(error);

                URL.revokeObjectURL(
                    objectURL
                );

                showToast(
                    "Could not load PLY model."
                );

            }
        );


        return;

    }


    if (extension === "obj") {

        const loader =
            new THREE.OBJLoader();


        loader.load(
            objectURL,
            object => {

                externalModelGroup.add(
                    object
                );

                setViewMode(
                    "digital"
                );

                fitView();

                URL.revokeObjectURL(
                    objectURL
                );


                logMessage(
                    "MODEL",
                    `${file.name} loaded.`
                );

                showToast(
                    "OBJ model loaded."
                );

            },

            undefined,

            error => {

                console.error(error);

                URL.revokeObjectURL(
                    objectURL
                );

                showToast(
                    "Could not load OBJ model."
                );

            }
        );


        return;

    }


    if (
        extension === "glb" ||
        extension === "gltf"
    ) {

        const loader =
            new THREE.GLTFLoader();


        loader.load(
            objectURL,
            gltf => {

                externalModelGroup.add(
                    gltf.scene
                );

                setViewMode(
                    "digital"
                );

                fitView();

                URL.revokeObjectURL(
                    objectURL
                );


                logMessage(
                    "MODEL",
                    `${file.name} loaded.`
                );

                showToast(
                    "GLTF model loaded."
                );

            },

            undefined,

            error => {

                console.error(error);

                URL.revokeObjectURL(
                    objectURL
                );

                showToast(
                    "Could not load GLTF model."
                );

            }
        );


        return;

    }


    showToast(
        "Unsupported 3D format."
    );

}


function clearExternalModel() {

    if (!externalModelGroup) return;


    while (
        externalModelGroup.children.length
    ) {

        const object =
            externalModelGroup.children[0];


        externalModelGroup.remove(
            object
        );


        object.traverse(
            child => {

                if (child.geometry) {
                    child.geometry.dispose();
                }

                if (child.material) {

                    if (
                        Array.isArray(
                            child.material
                        )
                    ) {

                        child.material.forEach(
                            material =>
                                material.dispose()
                        );

                    } else {

                        child.material.dispose();

                    }

                }

            }
        );

    }

}


/* ================= REFRESH ================= */

function refreshPipeline() {

    showToast(
        "Refreshing mission pipeline..."
    );


    logMessage(
        "SYSTEM",
        "Pipeline refresh requested."
    );


    loadReconstruction();

    loadTrajectory();

    detectSensors();

}


/* ================= MODULE STATUS ================= */

function setModuleStatus(
    id,
    text
) {

    const element =
        document.getElementById(id);


    if (!element) return;


    element.innerHTML =
        `<span></span> ${text}`;

}


/* ================= TEXT HELPERS ================= */

function setText(
    id,
    value
) {

    const element =
        document.getElementById(id);


    if (!element) return;


    element.textContent =
        value;

}


function formatNumber(number) {

    return Number(
        number
    ).toLocaleString(
        "en-US"
    );

}


/* ================= LOADING ================= */

function hideViewerLoading() {

    const element =
        document.getElementById(
            "viewerLoading"
        );


    if (!element) return;


    element.classList.add(
        "hidden"
    );

}


/* ================= LOGGING ================= */

function logMessage(
    category,
    message
) {

    const windowElement =
        document.getElementById(
            "logWindow"
        );


    if (!windowElement) return;


    const time =
        new Date()
            .toLocaleTimeString(
                [],
                {
                    hour12: false
                }
            );


    const entry =
        document.createElement(
            "div"
        );


    entry.className =
        "log-entry";


    entry.innerHTML = `
        <span class="log-time">
            ${time}
        </span>

        <span class="log-message">
            [${category}] ${message}
        </span>
    `;


    windowElement.appendChild(
        entry
    );


    windowElement.scrollTop =
        windowElement.scrollHeight;

}


function clearLogs() {

    const windowElement =
        document.getElementById(
            "logWindow"
        );


    windowElement.innerHTML = "";


    logMessage(
        "SYSTEM",
        "Activity log cleared."
    );

}


/* ================= TOAST ================= */

function showToast(
    message
) {

    const toast =
        document.getElementById(
            "toast"
        );

    const messageElement =
        document.getElementById(
            "toastMessage"
        );


    if (!toast || !messageElement) {
        return;
    }


    messageElement.textContent =
        message;


    toast.classList.add(
        "show"
    );


    clearTimeout(
        toastTimer
    );


    toastTimer =
        setTimeout(
            () => {

                toast.classList.remove(
                    "show"
                );

            },
            3000
        );

}
