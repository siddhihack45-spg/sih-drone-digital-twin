/* =========================================================
   PINKVISION AI
   FRONTEND CONTROLLER
========================================================= */


/* =========================================================
   GLOBAL STATE
========================================================= */

let scene;
let camera;
let renderer;
let controls;

let pointCloud = null;
let loadedModel = null;

let currentViewerMode = "pointcloud";

let animationFrameId = null;


/* =========================================================
   DOM HELPERS
========================================================= */

function $(selector) {
    return document.querySelector(selector);
}

function $$(selector) {
    return document.querySelectorAll(selector);
}


/* =========================================================
   INITIALIZATION
========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    initNavigation();

    init3DViewer();

    initViewerControls();

    initModelUpload();

    showToast(
        "PinkVision AI",
        "Geospatial reconstruction dashboard ready."
    );

});


/* =========================================================
   NAVIGATION
========================================================= */

function initNavigation() {

    const navItems = $$(".nav-item");

    navItems.forEach(item => {

        item.addEventListener("click", () => {

            const target = item.dataset.section;

            if (!target) return;

            navItems.forEach(nav => {
                nav.classList.remove("active");
            });

            item.classList.add("active");

            showSection(target);

        });

    });

}


function showSection(sectionId) {

    const sections = $$(".page-section");

    sections.forEach(section => {
        section.classList.remove("active");
    });

    const target = document.getElementById(sectionId);

    if (target) {
        target.classList.add("active");
    }


    const titles = {

        dashboard:
            "Geospatial Reconstruction Dashboard",

        reconstruction:
            "Single-Pass 3D Reconstruction",

        slam:
            "Visual SLAM & Trajectory",

        fusion:
            "Multi-Sensor Fusion",

        "digital-twin":
            "Georeferenced Digital Twin",

        analytics:
            "Reconstruction Analytics",

        confidence:
            "Confidence-Aware Geometry"

    };


    const pageTitle = $("#pageTitle");

    if (pageTitle) {
        pageTitle.textContent =
            titles[sectionId] || "PinkVision AI";
    }


    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });

}


/* =========================================================
   THREE.JS VIEWER
========================================================= */

function init3DViewer() {

    const container = $("#three-container");

    if (!container) return;


    scene = new THREE.Scene();

    scene.background = new THREE.Color(0xfffafc);


    /* Camera */

    camera = new THREE.PerspectiveCamera(
        55,
        container.clientWidth / container.clientHeight,
        0.01,
        10000
    );

    camera.position.set(
        3,
        2.5,
        5
    );


    /* Renderer */

    renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true
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


    /* Controls */

    controls = new THREE.OrbitControls(
        camera,
        renderer.domElement
    );

    controls.enableDamping = true;

    controls.dampingFactor = 0.06;

    controls.minDistance = 0.3;

    controls.maxDistance = 1000;


    /* Lighting */

    const ambientLight =
        new THREE.AmbientLight(
            0xffffff,
            1.4
        );

    scene.add(ambientLight);


    const directionalLight =
        new THREE.DirectionalLight(
            0xffffff,
            1.2
        );

    directionalLight.position.set(
        5,
        8,
        5
    );

    scene.add(directionalLight);


    /* Ground reference */

    createReferenceGrid();


    /* Initial model */

    createFallbackPointCloud();


    /* Resize */

    window.addEventListener(
        "resize",
        resizeViewer
    );


    /* Animation */

    animate();


    /* Load real reconstruction */

    loadPLYModel();

}


/* =========================================================
   REFERENCE GRID
========================================================= */

function createReferenceGrid() {

    const grid = new THREE.GridHelper(
        10,
        20,
        0xf0b6c9,
        0xf7dce6
    );

    grid.material.transparent = true;

    grid.material.opacity = 0.25;

    scene.add(grid);

}


/* =========================================================
   FALLBACK POINT CLOUD
========================================================= */

function createFallbackPointCloud() {

    if (!scene) return;


    const count = 1500;

    const positions =
        new Float32Array(count * 3);

    const colors =
        new Float32Array(count * 3);


    for (let i = 0; i < count; i++) {

        const angle =
            Math.random() * Math.PI * 2;

        const radius =
            Math.random() * 4;

        const x =
            Math.cos(angle) *
            radius;

        const y =
            (Math.random() - 0.5) *
            2;

        const z =
            Math.sin(angle) *
            radius;


        positions[i * 3] =
            x;

        positions[i * 3 + 1] =
            y;

        positions[i * 3 + 2] =
            z;


        colors[i * 3] =
            1.0;

        colors[i * 3 + 1] =
            0.45 + Math.random() * 0.3;

        colors[i * 3 + 2] =
            0.65 + Math.random() * 0.25;

    }


    const geometry =
        new THREE.BufferGeometry();

    geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(
            positions,
            3
        )
    );

    geometry.setAttribute(
        "color",
        new THREE.BufferAttribute(
            colors,
            3
        )
    );


    const material =
        new THREE.PointsMaterial({

            size: 0.035,

            vertexColors: true,

            transparent: true,

            opacity: 0.75,

            sizeAttenuation: true

        });


    pointCloud =
        new THREE.Points(
            geometry,
            material
        );

    scene.add(pointCloud);


    updatePointCount(count);

}


/* =========================================================
   REAL PLY LOADER
========================================================= */

function loadPLYModel() {

    if (!scene || !THREE.PLYLoader) {

        showToast(
            "Viewer Error",
            "PLY loader is not available."
        );

        return;
    }


    const loader =
        new THREE.PLYLoader();


    const path =
        "../output/reconstruction.ply";


    showViewerLoading(
        true,
        "Loading reconstruction.ply..."
    );


    loader.load(

        path,

        function(geometry) {

            try {

                /* Remove old model */

                if (loadedModel) {

                    scene.remove(
                        loadedModel
                    );

                    disposeObject(
                        loadedModel
                    );

                    loadedModel = null;

                }


                if (pointCloud) {

                    scene.remove(
                        pointCloud
                    );

                    disposeObject(
                        pointCloud
                    );

                    pointCloud = null;

                }


                /* Normals */

                if (
                    !geometry.attributes.normal
                ) {

                    geometry.computeVertexNormals();

                }


                /* Bounding box */

                geometry.computeBoundingBox();


                /* Colors */

                const hasColors =
                    !!geometry.attributes.color;


                const material =
                    new THREE.PointsMaterial({

                        size: calculatePointSize(
                            geometry
                        ),

                        vertexColors:
                            hasColors,

                        color:
                            hasColors
                                ? 0xffffff
                                : 0xf26f9d,

                        transparent: true,

                        opacity: 0.88,

                        sizeAttenuation: true

                    });


                loadedModel =
                    new THREE.Points(
                        geometry,
                        material
                    );


                /* Center model */

                centerObject(
                    loadedModel,
                    geometry
                );


                scene.add(
                    loadedModel
                );


                /* Point count */

                const count =
                    geometry.attributes
                        .position
                        .count;

                updatePointCount(
                    count
                );


                /* Camera */

                fitCameraToObject(
                    loadedModel
                );


                currentViewerMode =
                    "pointcloud";


                showViewerLoading(
                    false
                );


                showToast(
                    "3D Twin Loaded",
                    formatNumber(count) +
                    " points loaded from reconstruction.ply."
                );


            } catch (error) {

                console.error(error);

                showViewerLoading(
                    false
                );

                showToast(
                    "Model Error",
                    "Could not process the PLY geometry."
                );

            }

        },

        function(xhr) {

            if (
                xhr &&
                xhr.lengthComputable
            ) {

                const percent =
                    Math.round(
                        (
                            xhr.loaded /
                            xhr.total
                        ) * 100
                    );

                showViewerLoading(
                    true,
                    `Loading 3D model... ${percent}%`
                );

            }

        },

        function(error) {

            console.error(
                "PLY loading error:",
                error
            );


            showViewerLoading(
                false
            );


            showToast(
                "Model Not Found",
                "Using preview geometry. Check output/reconstruction.ply."
            );

        }

    );

}


/* =========================================================
   CENTER OBJECT
========================================================= */

function centerObject(
    object,
    geometry
) {

    if (!geometry.boundingBox) {

        geometry.computeBoundingBox();

    }


    const center =
        new THREE.Vector3();

    geometry.boundingBox.getCenter(
        center
    );


    object.position.sub(
        center
    );

}


/* =========================================================
   CAMERA FIT
========================================================= */

function fitCameraToObject(
    object
) {

    const box =
        new THREE.Box3()
            .setFromObject(object);


    const size =
        box.getSize(
            new THREE.Vector3()
        );

    const center =
        box.getCenter(
            new THREE.Vector3()
        );


    const maxDim =
        Math.max(
            size.x,
            size.y,
            size.z
        );


    const distance =
        Math.max(
            maxDim * 1.7,
            2
        );


    camera.position.set(
        center.x + distance,
        center.y + distance * 0.65,
        center.z + distance
    );


    camera.lookAt(
        center
    );


    controls.target.copy(
        center
    );


    controls.update();

}


/* =========================================================
   POINT SIZE
========================================================= */

function calculatePointSize(
    geometry
) {

    if (!geometry.boundingBox) {

        geometry.computeBoundingBox();

    }


    const size =
        geometry.boundingBox.getSize(
            new THREE.Vector3()
        );


    const maxDim =
        Math.max(
            size.x,
            size.y,
            size.z
        );


    if (maxDim > 100) {
        return 0.8;
    }

    if (maxDim > 20) {
        return 0.18;
    }

    if (maxDim > 5) {
        return 0.07;
    }

    return 0.035;

}


/* =========================================================
   MODEL UPLOAD
========================================================= */

function initModelUpload() {

    const input =
        $("#modelUpload");

    if (!input) return;


    input.addEventListener(
        "change",
        event => {

            const file =
                event.target.files[0];

            if (!file) return;


            const extension =
                file.name
                    .split(".")
                    .pop()
                    .toLowerCase();


            if (extension === "ply") {

                loadUploadedPLY(
                    file
                );

            } else {

                showToast(
                    "Upload Ready",
                    `${file.name} selected. PLY loading is currently enabled.`
                );

            }

        }
    );

}


/* =========================================================
   UPLOADED PLY
========================================================= */

function loadUploadedPLY(file) {

    const loader =
        new THREE.PLYLoader();


    const reader =
        new FileReader();


    reader.onload = function(event) {

        try {

            const geometry =
                loader.parse(
                    event.target.result
                );


            if (!geometry.attributes.normal) {

                geometry.computeVertexNormals();

            }


            geometry.computeBoundingBox();


            if (loadedModel) {

                scene.remove(
                    loadedModel
                );

                disposeObject(
                    loadedModel
                );

            }


            if (pointCloud) {

                scene.remove(
                    pointCloud
                );

                disposeObject(
                    pointCloud
                );

            }


            const material =
                new THREE.PointsMaterial({

                    size:
                        calculatePointSize(
                            geometry
                        ),

                    vertexColors:
                        !!geometry.attributes.color,

                    color: 0xf26f9d,

                    sizeAttenuation: true,

                    transparent: true,

                    opacity: 0.9

                });


            loadedModel =
                new THREE.Points(
                    geometry,
                    material
                );


            centerObject(
                loadedModel,
                geometry
            );


            scene.add(
                loadedModel
            );


            const count =
                geometry.attributes
                    .position
                    .count;


            updatePointCount(
                count
            );


            fitCameraToObject(
                loadedModel
            );


            showToast(
                "Model Loaded",
                `${file.name} contains ${formatNumber(count)} points.`
            );


        } catch (error) {

            console.error(error);

            showToast(
                "Upload Error",
                "Unable to parse this PLY file."
            );

        }

    };


    reader.readAsArrayBuffer(
        file
    );

}


/* =========================================================
   VIEWER CONTROLS
========================================================= */

function initViewerControls() {

    const buttons =
        $$(".viewer-mode");


    buttons.forEach(button => {

        button.addEventListener(
            "click",
            () => {

                buttons.forEach(
                    item =>
                        item.classList.remove(
                            "active"
                        )
                );


                button.classList.add(
                    "active"
                );


                const mode =
                    button.dataset.mode;


                setViewerMode(
                    mode
                );

            }
        );

    });

}


function setViewerMode(mode) {

    currentViewerMode =
        mode;


    const object =
        loadedModel ||
        pointCloud;


    if (!object) return;


    if (!object.material) return;


    if (mode === "points") {

        object.material.size =
            0.06;

        object.material.opacity =
            1.0;

    }


    if (mode === "pointcloud") {

        object.material.size =
            calculatePointSize(
                object.geometry
            );

        object.material.opacity =
            0.88;

    }


    if (mode === "wireframe") {

        object.material.size =
            0.02;

        object.material.opacity =
            0.45;

    }


    object.material.needsUpdate =
        true;


    showToast(
        "Viewer Mode",
        `${mode} visualization enabled.`
    );

}


/* =========================================================
   ANIMATION
========================================================= */

function animate() {

    animationFrameId =
        requestAnimationFrame(
            animate
        );


    if (controls) {

        controls.update();

    }


    if (
        loadedModel &&
        currentViewerMode === "pointcloud"
    ) {

        loadedModel.rotation.y +=
            0.0003;

    }


    if (
        pointCloud &&
        !loadedModel
    ) {

        pointCloud.rotation.y +=
            0.0005;

    }


    if (renderer && scene && camera) {

        renderer.render(
            scene,
            camera
        );

    }

}


/* =========================================================
   RESIZE
========================================================= */

function resizeViewer() {

    const container =
        $("#three-container");

    if (
        !container ||
        !camera ||
        !renderer
    ) {

        return;

    }


    camera.aspect =
        container.clientWidth /
        container.clientHeight;


    camera.updateProjectionMatrix();


    renderer.setSize(
        container.clientWidth,
        container.clientHeight
    );

}


/* =========================================================
   PIPELINE
========================================================= */

function startPipeline() {

    showToast(
        "Pipeline Started",
        "Running single-pass reconstruction pipeline..."
    );


    const steps =
        document.querySelectorAll(
            ".pipeline-step"
        );


    steps.forEach(
        step =>
            step.classList.remove(
                "active"
            )
    );


    let index = 0;


    const interval =
        setInterval(() => {

            if (index >= steps.length) {

                clearInterval(
                    interval
                );


                showToast(
                    "Pipeline Ready",
                    "Reconstruction pipeline completed."
                );


                return;

            }


            steps[index]
                .classList.add(
                    "active"
                );


            index++;

        }, 500);

}


/* =========================================================
   TRAJECTORY
========================================================= */

function openTrajectory() {

    const nav =
        document.querySelector(
            '[data-section="slam"]'
        );


    if (nav) {

        nav.click();

    }

}


/* =========================================================
   POINT COUNT
========================================================= */

function updatePointCount(
    count
) {

    const element =
        $("#pointCount");


    if (!element) return;


    element.textContent =
        formatNumber(count);

}


function formatNumber(
    number
) {

    return Number(number || 0)
        .toLocaleString(
            "en-IN"
        );

}


/* =========================================================
   LOADING
========================================================= */

function showViewerLoading(
    visible,
    message
) {

    const loading =
        $("#viewerLoading");


    if (!loading) return;


    if (message) {

        const text =
            loading.querySelector(
                "span"
            );

        if (text) {

            text.textContent =
                message;

        }

    }


    loading.style.opacity =
        visible ? "1" : "0";

}


/* =========================================================
   TOAST
========================================================= */

let toastTimer = null;


function showToast(
    title,
    message
) {

    const toast =
        $("#toast");

    const titleElement =
        $("#toastTitle");

    const messageElement =
        $("#toastMessage");


    if (
        !toast ||
        !titleElement ||
        !messageElement
    ) {

        return;

    }


    titleElement.textContent =
        title;

    messageElement.textContent =
        message;


    toast.classList.add(
        "show"
    );


    clearTimeout(
        toastTimer
    );


    toastTimer =
        setTimeout(() => {

            toast.classList.remove(
                "show"
            );

        }, 3500);

}


/* =========================================================
   DISPOSE THREE OBJECT
========================================================= */

function disposeObject(
    object
) {

    if (!object) return;


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


/* =========================================================
   KEYBOARD SHORTCUTS
========================================================= */

document.addEventListener(
    "keydown",
    event => {

        if (
            event.ctrlKey &&
            event.key.toLowerCase() === "r"
        ) {

            event.preventDefault();

            startPipeline();

        }

    }
);