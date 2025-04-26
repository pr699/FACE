 // Matrix rain effect
 const matrixCanvas = document.getElementById('matrix-rain');
 const matrixCtx = matrixCanvas.getContext('2d');
 
 function resizeMatrixCanvas() {
     matrixCanvas.width = document.documentElement.scrollWidth;
     matrixCanvas.height = document.documentElement.scrollHeight;
 }
 
 resizeMatrixCanvas();
 
 const matrixChars = "01";
 const fontSize = 14;
 const columns = matrixCanvas.width / fontSize;
 
 const drops = [];
 for (let i = 0; i < columns; i++) {
     drops[i] = Math.random() * -100;
 }
 
 function drawMatrix() {
     matrixCtx.fillStyle = 'rgba(0, 0, 0, 0.05)';
     matrixCtx.fillRect(0, 0, matrixCanvas.width, matrixCanvas.height);
     
     matrixCtx.fillStyle = '#0F0';
     matrixCtx.font = fontSize + 'px monospace';
     
     for (let i = 0; i < drops.length; i++) {
         const text = matrixChars.charAt(Math.floor(Math.random() * matrixChars.length));
         matrixCtx.fillText(text, i * fontSize, drops[i] * fontSize);
         
         if (drops[i] * fontSize > matrixCanvas.height && Math.random() > 0.975) {
             drops[i] = 0;
         }
         
         drops[i]++;
     }
 }
 
 window.addEventListener('resize', () => {
     resizeMatrixCanvas();
 });
 
 setInterval(drawMatrix, 33);

 // Face recognition code
 const video = document.getElementById('webcam');
 const startButton = document.getElementById('startButton');
 const stopButton = document.getElementById('stopButton');
 const switchCameraBtn = document.getElementById('switchCameraBtn');
 const statusDiv = document.getElementById('status');
 const overlay = document.getElementById('overlay');
 const fileInput = document.getElementById('fileInput');
 const labelInput = document.getElementById('labelInput');
 const addTrainingBtn = document.getElementById('addTrainingBtn');
 const finalTrainBtn = document.getElementById('finalTrainBtn');
 const trainingSetsDiv = document.getElementById('trainingSets');
 const labelsListDiv = document.getElementById('labelsList');
 
 let stream = null;
 let modelsLoaded = false;
 let faceMatcher = null;
 let trainingData = [];
 let pendingTrainingSets = [];
 let currentFacingMode = "user";
 let isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
 let detectionInProgress = false;

 function supportsMultipleCameras() {
     return navigator.mediaDevices && navigator.mediaDevices.enumerateDevices && isMobile;
 }

 async function loadModels() {
     statusDiv.textContent = "BOOTING-UP...";
     
     try {
         await Promise.all([
             faceapi.nets.tinyFaceDetector.loadFromUri('https://justadudewhohacks.github.io/face-api.js/models'),
             faceapi.nets.faceLandmark68Net.loadFromUri('https://justadudewhohacks.github.io/face-api.js/models'),
             faceapi.nets.faceRecognitionNet.loadFromUri('https://justadudewhohacks.github.io/face-api.js/models')
         ]);
         
         modelsLoaded = true;
         statusDiv.textContent = "LOADING COMPLETED. AWAITING TRAINING DATA...";
         
         if (supportsMultipleCameras()) {
             try {
                 const devices = await navigator.mediaDevices.enumerateDevices();
                 const videoDevices = devices.filter(device => device.kind === 'videoinput');
                 if (videoDevices.length > 1) {
                     switchCameraBtn.disabled = false;
                 }
             } catch (err) {
                 console.log("Camera enumeration not supported:", err);
             }
         }
         
     } catch (err) {
         console.error("Error loading models:", err);
         statusDiv.textContent = "ERROR: MODEL LOAD FAILURE";
     }
 }

 addTrainingBtn.addEventListener('click', () => {
     const label = labelInput.value.trim();
     const files = fileInput.files;
     
     if (!label || files.length === 0) {
         statusDiv.textContent = "ERROR: PROVIDE NAME AND IMAGES";
         return;
     }
     
     pendingTrainingSets.push({
         label,
         files: Array.from(files)
     });
     
     const setDiv = document.createElement('div');
     setDiv.innerHTML = `
         <h4>${label} (${files.length} IMAGES)</h4>
         <div class="previews"></div>
     `;
     
     const previewsDiv = setDiv.querySelector('.previews');
     Array.from(files).forEach(file => {
         const img = document.createElement('img');
         img.className = 'image-preview';
         img.src = URL.createObjectURL(file);
         img.onload = () => URL.revokeObjectURL(img.src);
         previewsDiv.appendChild(img);
     });
     
     trainingSetsDiv.appendChild(setDiv);
     updateLabelsList();
     labelInput.value = '';
     fileInput.value = '';
     finalTrainBtn.disabled = false;
     
     statusDiv.textContent = `TRAINING SET ADDED: ${label}`;
 });

 function updateLabelsList() {
     labelsListDiv.innerHTML = '';
     const uniqueLabels = [...new Set(pendingTrainingSets.map(set => set.label))];
     
     if (uniqueLabels.length === 0) {
         labelsListDiv.innerHTML = '<p>NO TRAINING DATA</p>';
         return;
     }
     
     const list = document.createElement('ul');
     uniqueLabels.forEach(label => {
         const count = pendingTrainingSets.filter(set => set.label === label).length;
         const item = document.createElement('li');
         item.textContent = `${label} (${count} SET${count > 1 ? 'S' : ''})`;
         list.appendChild(item);
     });
     
     labelsListDiv.appendChild(list);
 }

 finalTrainBtn.addEventListener('click', async () => {
     if (pendingTrainingSets.length === 0) return;
     
     statusDiv.textContent = "PROCESSING TRAINING DATA...";
     finalTrainBtn.disabled = true;
     
     try {
         const labeledDescriptors = [];
         
         for (const set of pendingTrainingSets) {
             const descriptors = [];
             
             for (const file of set.files) {
                 const img = await faceapi.bufferToImage(file);
                 const detections = await faceapi
                     .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
                     .withFaceLandmarks()
                     .withFaceDescriptors();
                 
                 if (detections.length > 0) {
                     descriptors.push(detections[0].descriptor);
                 }
             }
             
             if (descriptors.length > 0) {
                 labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(set.label, descriptors));
             }
         }
         
         if (labeledDescriptors.length > 0) {
             faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
             trainingData = [...pendingTrainingSets];
             pendingTrainingSets = [];
             
             statusDiv.textContent = `TRAINING COMPLETE: ${labeledDescriptors.length} SUBJECTS`;
             startButton.disabled = false;
             trainingSetsDiv.innerHTML = '';
             updateLabelsList();
             finalTrainBtn.disabled = true;
         } else {
             statusDiv.textContent = "ERROR: NO FACES DETECTED IN TRAINING DATA";
             finalTrainBtn.disabled = false;
         }
         
     } catch (err) {
         console.error("Training error:", err);
         statusDiv.textContent = "ERROR: TRAINING FAILURE";
         finalTrainBtn.disabled = false;
     }
 });

 async function detectFaces() {
     if (detectionInProgress) return;
     
     detectionInProgress = true;
     
     try {
         const detections = await faceapi
             .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
             .withFaceLandmarks()
             .withFaceDescriptors();
         
         const canvas = document.getElementById('overlay');
         const context = canvas.getContext('2d');
         context.clearRect(0, 0, canvas.width, canvas.height);
         
         const displaySize = { 
             width: video.videoWidth, 
             height: video.videoHeight 
         };
         
         const resizedDetections = faceapi.resizeResults(detections, displaySize);
         
         const isFrontCamera = currentFacingMode === "user";
         
         resizedDetections.forEach(detection => {
             const box = detection.detection.box;
             let label = "UNKNOWN";
             let color = "#FF0000";
             
             if (faceMatcher) {
                 const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
                 if (bestMatch.distance < 0.6) {
                     label = bestMatch.label;
                     color = "#0F0";
                 }
             }
             
             let boxX = box.x;
             if (isFrontCamera) {
                 boxX = displaySize.width - box.x - box.width;
             }
             
             context.strokeStyle = color;
             context.lineWidth = 2;
             context.strokeRect(boxX, box.y, box.width, box.height);
             
             context.fillStyle = "rgba(0, 0, 0, 0.7)";
             const textWidth = context.measureText(label).width;
             context.fillRect(boxX, box.y - 20, textWidth + 10, 20);
             
             context.fillStyle = color;
             context.font = "bold 14px 'Courier New'";
             context.fillText(label, boxX + 5, box.y - 5);
         });
     } catch (err) {
         console.error("Face detection error:", err);
     } finally {
         detectionInProgress = false;
         setTimeout(detectFaces, 33);
     }
 }

 function startFaceRecognition() {
     if (!modelsLoaded) return;
     
     overlay.width = video.videoWidth;
     overlay.height = video.videoHeight;
     
     detectFaces();
 }

 function stopFaceRecognition() {
     detectionInProgress = false;
     const context = overlay.getContext('2d');
     context.clearRect(0, 0, overlay.width, overlay.height);
 }

 async function startCamera(facingMode = "user") {
     try {
         if (stream) {
             stream.getTracks().forEach(track => track.stop());
         }
         
         const constraints = {
             video: {
                 width: { ideal: 640 },
                 height: { ideal: 480 },
                 facingMode: facingMode
             },
             audio: false
         };
         
         if (/iPhone|iPad|iPod/i.test(navigator.userAgent)) {
             delete constraints.video.width;
             delete constraints.video.height;
         }
         
         stream = await navigator.mediaDevices.getUserMedia(constraints);
         
         video.srcObject = stream;
         video.style.display = 'block';
         startButton.disabled = true;
         stopButton.disabled = false;
         
         // Mirror only front camera
         video.style.transform = facingMode === "user" ? "rotateY(180deg)" : "none";
         
         if (isMobile) {
             video.setAttribute('playsinline', '');
             video.setAttribute('webkit-playsinline', '');
         }
         
         return new Promise((resolve) => {
             video.onplaying = () => {
                 overlay.width = video.videoWidth;
                 overlay.height = video.videoHeight;
                 
                 startFaceRecognition();
                 resolve();
             };
         });
         
     } catch (err) {
         console.error("Error accessing camera:", err);
         statusDiv.textContent = "ERROR: CAMERA ACCESS DENIED";
         throw err;
     }
 }

 startButton.addEventListener('click', async () => {
     try {
         await startCamera(currentFacingMode);
     } catch (err) {
         startButton.disabled = false;
         stopButton.disabled = true;
     }
 });

 stopButton.addEventListener('click', () => {
     stopFaceRecognition();
     
     if (stream) {
         stream.getTracks().forEach(track => track.stop());
         video.srcObject = null;
         video.style.display = 'none';
         startButton.disabled = false;
         stopButton.disabled = true;
         
         statusDiv.textContent = "CAMERA FEED TERMINATED";
     }
 });

 switchCameraBtn.addEventListener('click', async () => {
     if (!stream) return;
     
     try {
         switchCameraBtn.disabled = true;
         statusDiv.textContent = "BACK CAMERA ONLINE";
         
         currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
         
         await startCamera(currentFacingMode);
         
         switchCameraBtn.disabled = false;
     } catch (err) {
         console.error("Error switching camera:", err);
         statusDiv.textContent = "ERROR: FAILED TO SWITCH CAMERA";
         switchCameraBtn.disabled = false;
     }
 });

 loadModels();
 
 if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
     statusDiv.textContent = "ERROR: BROWSER NOT SUPPORTED";
     startButton.disabled = true;
     switchCameraBtn.disabled = true;
 }

 window.addEventListener('orientationchange', () => {
     if (stream) {
         setTimeout(() => {
             overlay.width = video.videoWidth;
             overlay.height = video.videoHeight;
         }, 500);
     }
 });