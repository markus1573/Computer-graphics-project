// WebGPU Boeing 747 Model Viewer
// Uses OBJParser.js and MV.js for model loading and matrix operations

// Global variables
let device, context, pipeline;
let uniformBuffer;
let uniformBindGroupBody, uniformBindGroupAileron, uniformBindGroupRAileron;
let uniformBindGroupLElevator, uniformBindGroupRElevator;
let uniformBindGroupRudder;

// Textures for MSAA and Depth
let depthTexture, msaaTexture;

// Body buffers
let bodyVertexBuffer, bodyNormalBuffer, bodyColorBuffer, bodyIndexBuffer;
let numBodyIndices = 0;

// Left Aileron buffers
let aileronVertexBuffer, aileronNormalBuffer, aileronColorBuffer, aileronIndexBuffer;
let numAileronIndices = 0;

// Right Aileron buffers
let rAileronVertexBuffer, rAileronNormalBuffer, rAileronColorBuffer, rAileronIndexBuffer;
let numRAileronIndices = 0;

// Left Elevator buffers
let lElevatorVertexBuffer, lElevatorNormalBuffer, lElevatorColorBuffer, lElevatorIndexBuffer;
let numLElevatorIndices = 0;

// Right Elevator buffers
let rElevatorVertexBuffer, rElevatorNormalBuffer, rElevatorColorBuffer, rElevatorIndexBuffer;
let numRElevatorIndices = 0;

// Rudder buffers
let rudderVertexBuffer, rudderNormalBuffer, rudderColorBuffer, rudderIndexBuffer;
let numRudderIndices = 0;

let modelViewMatrix, projectionMatrix;
let autoRotate = false;
let rotationStartTime = Date.now();
let rotationOffset = Math.PI; // Start from behind (180 degrees)
let modelLoaded = false;

// Animation state
let aileronAngle = 0;
let rAileronAngle = 0;
let elevatorAngle = 0;
let rudderAngle = 0;
const AILERON_MAX_ANGLE = 45;

// Camera parameters
let eye = vec3(0, 5, -20);
let at = vec3(0, 0, 0);
let up = vec3(0, 1, 0);

// WGSL Shader code
const shaderCode = `
struct Uniforms {
    modelViewMatrix: mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,
    normalMatrix: mat4x4<f32>,
    lightDirection: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) viewPosition: vec3<f32>,
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    
    // Transform position
    let modelViewPos = uniforms.modelViewMatrix * input.position;
    output.position = uniforms.projectionMatrix * modelViewPos;
    output.viewPosition = modelViewPos.xyz;
    
    // Transform normal to view space
    let transformedNormal = uniforms.normalMatrix * input.normal;
    output.normal = normalize(transformedNormal.xyz);
    
    output.color = input.color;
    
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    // Re-normalize interpolated vectors
    let N = normalize(input.normal);
    let V = normalize(-input.viewPosition);
    
    // Lighting parameters
    let kd = 1.0;
    let ks = 0.5;
    let shininess = 250.0;
    let ka = 1.0;
    
    let le = 1.0;
    let la = 0.2;
    
    // Light direction (Sun light, transformed to View Space)
    let L = normalize(uniforms.lightDirection.xyz);
    
    let lightEmission = vec3<f32>(le, le, le);
    let ambientLight = vec3<f32>(la, la, la);
    let specularColor = vec3<f32>(1.0, 1.0, 1.0);
    
    let diffuseColor = input.color.rgb;
    
    // Ambient
    let ambient = ka * diffuseColor * ambientLight;
    
    // Diffuse
    let nDotL = max(dot(N, L), 0.0);
    let diffuse = kd * diffuseColor * lightEmission * nDotL;
    
    // Specular
    let reflectDir = reflect(-L, N);
    let rDotV = max(dot(reflectDir, V), 0.0);
    let specular = ks * specularColor * lightEmission * pow(rDotV, shininess);
    
    let finalColor = ambient + diffuse + specular;
    
    return vec4<f32>(finalColor, 1.0);
}
`;

// Initialize WebGPU
async function initWebGPU() {
    const canvas = document.getElementById('canvas');
    
    if (!navigator.gpu) {
        alert('WebGPU is not supported in your browser.');
        return false;
    }
    
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        alert('Failed to get GPU adapter');
        return false;
    }
    
    device = await adapter.requestDevice();
    
    context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
        alphaMode: 'premultiplied',
    });
    
    const shaderModule = device.createShaderModule({
        code: shaderCode
    });
    
    pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
            buffers: [
                {
                    arrayStride: 16,
                    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x4' }]
                },
                {
                    arrayStride: 16,
                    attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x4' }]
                },
                {
                    arrayStride: 16,
                    attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x4' }]
                }
            ]
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{ format: canvasFormat }]
        },
        primitive: {
            topology: 'triangle-list',
            cullMode: 'back',
            frontFace: 'ccw'
        },
        multisample: {
            count: 4,
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus'
        }
    });
    
    // Create uniform buffer for 6 objects (Body + L_Ail + R_Ail + L_Elev + R_Elev + Rudder)
    // Each object needs 208 bytes (3*64 + 16).
    // Min offset alignment is usually 256 bytes.
    const uniformBufferSize = 256 * 6; 
    uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    // Create bind groups for each object pointing to different offsets
    const bindGroupLayout = pipeline.getBindGroupLayout(0);
    
    uniformBindGroupBody = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: {
                buffer: uniformBuffer,
                offset: 0,
                size: 208 // Size of Uniforms struct (3*64 + 16)
            }
        }]
    });
    
    uniformBindGroupAileron = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: {
                buffer: uniformBuffer,
                offset: 256, // Start of second block
                size: 208
            }
        }]
    });

    uniformBindGroupRAileron = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: {
                buffer: uniformBuffer,
                offset: 512, // Start of third block
                size: 208
            }
        }]
    });

    uniformBindGroupLElevator = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: {
                buffer: uniformBuffer,
                offset: 768, // Start of fourth block
                size: 208
            }
        }]
    });

    uniformBindGroupRElevator = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: {
                buffer: uniformBuffer,
                offset: 1024, // Start of fifth block
                size: 208
            }
        }]
    });

    uniformBindGroupRudder = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: {
                buffer: uniformBuffer,
                offset: 1280, // Start of sixth block
                size: 208
            }
        }]
    });
    
    updateMatrices();
    
    return true;
}

// Helper to calculate matrices for a moving part
function calculatePartMatrices(baseModelMatrix, viewMatrix, pivot, axis, angle) {
    let pivotTrans = translate(pivot[0], pivot[1], pivot[2]);
    let pivotRot = rotate(angle, axis);
    let pivotInvTrans = translate(-pivot[0], -pivot[1], -pivot[2]);
    
    let localTransform = mult(pivotTrans, mult(pivotRot, pivotInvTrans));
    let partModelMatrix = mult(baseModelMatrix, localTransform);
    let partModelView = mult(viewMatrix, partModelMatrix);
    let partNormalMat = normalMatrix(partModelView, false);
    
    return { modelView: partModelView, normalMat: partNormalMat };
}

// Update transformation matrices
function updateMatrices() {
    // 1. Camera & Global Rotation
    // Fixed camera relative to plane
    eye = vec3(0, 5, -20);
    at = vec3(0, 0, 0);
    up = vec3(0, 1, 0);

    const viewMatrix = lookAt(eye, at, up);
    
    const canvas = document.getElementById('canvas');
    const aspect = canvas.width / canvas.height;
    projectionMatrix = perspective(45, aspect, 0.1, 100.0);
    
    // 2. Base Model Transform (Center and Scale)
    // Center the fuselage at (0,0,0)
    // Fuselage Center approx: X=1.95, Y=-2.42, Z=-1.3
    // Scale: 2.0
    // Tx = -1.95 * 2 = -3.9
    // Ty = -(-2.42) * 2 = 4.84
    // Tz = -(-1.3) * 2 = 2.6
    const baseTrans = translate(-3.9, 4.84, 2.6);
    const baseScale = scalem(2.0, 2.0, 2.0);
    let baseModelMatrix = mult(baseTrans, baseScale);
    
    // --- BODY TRANSFORM ---
    let bodyModelView = mult(viewMatrix, baseModelMatrix);
    let bodyNormalMat = normalMatrix(bodyModelView, false); // Return mat4
    
    // --- LEFT AILERON ---
    const lAileronPivot = vec3(4.0313, -2.4272, -2.4330);
    const lAileronAxis = vec3(0.5075, 0.0511, -0.3055);
    const lAileronMatrices = calculatePartMatrices(baseModelMatrix, viewMatrix, lAileronPivot, lAileronAxis, aileronAngle);
    let lAileronModelView = lAileronMatrices.modelView;
    let lAileronNormalMat = lAileronMatrices.normalMat;

    // --- RIGHT AILERON ---
    const rAileronPivot = vec3(-0.0806, -2.4553, -2.4306);
    const rAileronAxis = vec3(-0.5087, 0.0386, -0.3047);
    const rAileronMatrices = calculatePartMatrices(baseModelMatrix, viewMatrix, rAileronPivot, rAileronAxis, rAileronAngle);
    let rAileronModelView = rAileronMatrices.modelView;
    let rAileronNormalMat = rAileronMatrices.normalMat;

    // --- LEFT ELEVATOR ---
    const lElevatorPivot = vec3(2.1539, -2.2897, -4.6862);
    const lElevatorAxis = vec3(0.7007, 0.1209, -0.2267);
    const lElevatorMatrices = calculatePartMatrices(baseModelMatrix, viewMatrix, lElevatorPivot, lElevatorAxis, elevatorAngle);
    let lElevatorModelView = lElevatorMatrices.modelView;
    let lElevatorNormalMat = lElevatorMatrices.normalMat;

    // --- RIGHT ELEVATOR ---
    const rElevatorPivot = vec3(1.7560, -2.2713, -4.6876);
    const rElevatorAxis = vec3(-0.7060, 0.0917, -0.2239);
    // Note: Inverted angle for right elevator
    const rElevatorMatrices = calculatePartMatrices(baseModelMatrix, viewMatrix, rElevatorPivot, rElevatorAxis, -elevatorAngle);
    let rElevatorModelView = rElevatorMatrices.modelView;
    let rElevatorNormalMat = rElevatorMatrices.normalMat;

    // --- RUDDER ---
    const rudderPivot = vec3(1.9563, -1.9577, -4.6281);
    const rudderAxis = vec3(-0.0078, 0.6877, -0.2899);
    const rudderMatrices = calculatePartMatrices(baseModelMatrix, viewMatrix, rudderPivot, rudderAxis, rudderAngle);
    let rudderModelView = rudderMatrices.modelView;
    let rudderNormalMat = rudderMatrices.normalMat;

    
    // --- LIGHT DIRECTION ---
    // Sun direction in World Space (e.g. from straight up)
    let sunDirWorld = vec4(0.0, 1.0, 0.0, 0.0); // Directional light, w=0
    
    // Rotate light opposite to plane rotation to simulate world rotation
    // Plane Roll (Z) = aileronAngle
    // Plane Pitch (X) = elevatorAngle
    // Plane Yaw (Y) = rudderAngle
    
    // Inverse rotations (negative angles)
    let rotZ = rotate(-aileronAngle, vec3(0, 0, 1));
    let rotX = rotate(-elevatorAngle, vec3(1, 0, 0));
    let rotY = rotate(-rudderAngle, vec3(0, 1, 0));
    
    // Apply rotations: Yaw -> Pitch -> Roll (inverse order of application for world)
    // Actually, just rotating the vector is enough.
    // Order: If plane rolls, world rolls opposite.
    let lightRot = mult(rotZ, mult(rotX, rotY));
    sunDirWorld = mult(lightRot, sunDirWorld);

    // Transform to View Space: L_view = View * L_world
    // Note: mult(mat4, vec4) returns array [x,y,z,w]
    let sunDirView = mult(viewMatrix, sunDirWorld);
    
    // --- UPDATE BACKGROUND CSS ---
    const bgImage = document.getElementById('background-image');
    if (bgImage) {
        // Map angles to transforms
        // Yaw (Y-axis rotation) -> X translation
        // Pitch (X-axis rotation) -> Y translation
        // Roll (Z-axis rotation) -> Rotation
        
        // Scale factors (tweak as needed)
        const xTrans = -rudderAngle * 5; // px
        const yTrans = elevatorAngle * 5; // px
        const rot = -aileronAngle; // deg (Roll left = background rotates right)
        
        bgImage.style.transform = `translate(${xTrans}px, ${yTrans}px) rotate(${rot}deg)`;
    }
    
    // --- WRITE TO BUFFER ---
    if (device && uniformBuffer) {
        // Body Data (Offset 0)
        const bodyData = new Float32Array([
            ...flatten(bodyModelView),
            ...flatten(projectionMatrix),
            ...flatten(bodyNormalMat),
            ...sunDirView
        ]);
        device.queue.writeBuffer(uniformBuffer, 0, bodyData);
        
        // Left Aileron Data (Offset 256)
        const aileronData = new Float32Array([
            ...flatten(lAileronModelView),
            ...flatten(projectionMatrix),
            ...flatten(lAileronNormalMat),
            ...sunDirView
        ]);
        device.queue.writeBuffer(uniformBuffer, 256, aileronData);

        // Right Aileron Data (Offset 512)
        const rAileronData = new Float32Array([
            ...flatten(rAileronModelView),
            ...flatten(projectionMatrix),
            ...flatten(rAileronNormalMat),
            ...sunDirView
        ]);
        device.queue.writeBuffer(uniformBuffer, 512, rAileronData);

        // Left Elevator Data (Offset 768)
        const lElevData = new Float32Array([
            ...flatten(lElevatorModelView),
            ...flatten(projectionMatrix),
            ...flatten(lElevatorNormalMat),
            ...sunDirView
        ]);
        device.queue.writeBuffer(uniformBuffer, 768, lElevData);

        // Right Elevator Data (Offset 1024)
        const rElevData = new Float32Array([
            ...flatten(rElevatorModelView),
            ...flatten(projectionMatrix),
            ...flatten(rElevatorNormalMat),
            ...sunDirView
        ]);
        device.queue.writeBuffer(uniformBuffer, 1024, rElevData);

        // Rudder Data (Offset 1280)
        const rudderData = new Float32Array([
            ...flatten(rudderModelView),
            ...flatten(projectionMatrix),
            ...flatten(rudderNormalMat),
            ...sunDirView
        ]);
        device.queue.writeBuffer(uniformBuffer, 1280, rudderData);
    }
}

// Helper to create buffers
function createMeshBuffers(drawingInfo) {
    const vBuffer = device.createBuffer({
        size: drawingInfo.vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(vBuffer, 0, drawingInfo.vertices);
    
    const nBuffer = device.createBuffer({
        size: drawingInfo.normals.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(nBuffer, 0, drawingInfo.normals);
    
    const cBuffer = device.createBuffer({
        size: drawingInfo.colors.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(cBuffer, 0, drawingInfo.colors);
    
    const iBuffer = device.createBuffer({
        size: drawingInfo.indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(iBuffer, 0, drawingInfo.indices);
    
    return { vBuffer, nBuffer, cBuffer, iBuffer };
}

// Load and parse OBJ file
async function loadModel() {
    const loadingStatus = document.getElementById('loadingStatus');
    const modelInfo = document.getElementById('modelInfo');
    loadingStatus.style.display = 'block';
    loadingStatus.textContent = 'Loading Boeing 747 model...';
    try {
        // Load the OBJ file
        const result = await readOBJFile('Boeing747_simple_v2.obj', 1.0, false);
        
        if (!result) {
            throw new Error('Failed to load OBJ file');
        }
        
        const doc = result.doc;
        
        // Split geometry
        // Exclude ailerons, elevators and rudder from body
        const bodyInfo = doc.getDrawingInfoExcludingGroups(['L_aileron_Cube', 'R_aileron_Cube.001', 'L_elevator_Cube.003', 'R_elevator_Cube.002', 'Rudder_Cube.004']);
        const aileronInfo = doc.getDrawingInfoForGroup('L_aileron_Cube');
        const rAileronInfo = doc.getDrawingInfoForGroup('R_aileron_Cube.001');
        const lElevatorInfo = doc.getDrawingInfoForGroup('L_elevator_Cube.003');
        const rElevatorInfo = doc.getDrawingInfoForGroup('R_elevator_Cube.002');
        const rudderInfo = doc.getDrawingInfoForGroup('Rudder_Cube.004');
        
        if (!bodyInfo || !aileronInfo || !rAileronInfo || !lElevatorInfo || !rElevatorInfo || !rudderInfo) {
             throw new Error('Failed to split model geometry. Check group names.');
        }
        
        // Create Body Buffers
        numBodyIndices = bodyInfo.indices.length;
        const bodyBuffers = createMeshBuffers(bodyInfo);
        bodyVertexBuffer = bodyBuffers.vBuffer;
        bodyNormalBuffer = bodyBuffers.nBuffer;
        bodyColorBuffer = bodyBuffers.cBuffer;
        bodyIndexBuffer = bodyBuffers.iBuffer;
        
        // Create Left Aileron Buffers
        numAileronIndices = aileronInfo.indices.length;
        const aileronBuffers = createMeshBuffers(aileronInfo);
        aileronVertexBuffer = aileronBuffers.vBuffer;
        aileronNormalBuffer = aileronBuffers.nBuffer;
        aileronColorBuffer = aileronBuffers.cBuffer;
        aileronIndexBuffer = aileronBuffers.iBuffer;

        // Create Right Aileron Buffers
        numRAileronIndices = rAileronInfo.indices.length;
        const rAileronBuffers = createMeshBuffers(rAileronInfo);
        rAileronVertexBuffer = rAileronBuffers.vBuffer;
        rAileronNormalBuffer = rAileronBuffers.nBuffer;
        rAileronColorBuffer = rAileronBuffers.cBuffer;
        rAileronIndexBuffer = rAileronBuffers.iBuffer;

        // Create Left Elevator Buffers
        numLElevatorIndices = lElevatorInfo.indices.length;
        const lElevBuffers = createMeshBuffers(lElevatorInfo);
        lElevatorVertexBuffer = lElevBuffers.vBuffer;
        lElevatorNormalBuffer = lElevBuffers.nBuffer;
        lElevatorColorBuffer = lElevBuffers.cBuffer;
        lElevatorIndexBuffer = lElevBuffers.iBuffer;

        // Create Right Elevator Buffers
        numRElevatorIndices = rElevatorInfo.indices.length;
        const rElevBuffers = createMeshBuffers(rElevatorInfo);
        rElevatorVertexBuffer = rElevBuffers.vBuffer;
        rElevatorNormalBuffer = rElevBuffers.nBuffer;
        rElevatorColorBuffer = rElevBuffers.cBuffer;
        rElevatorIndexBuffer = rElevBuffers.iBuffer;

        // Create Rudder Buffers
        numRudderIndices = rudderInfo.indices.length;
        const rudderBuffers = createMeshBuffers(rudderInfo);
        rudderVertexBuffer = rudderBuffers.vBuffer;
        rudderNormalBuffer = rudderBuffers.nBuffer;
        rudderColorBuffer = rudderBuffers.cBuffer;
        rudderIndexBuffer = rudderBuffers.iBuffer;
        
        // Display info
        modelInfo.innerHTML = `
            <p><strong>Model loaded successfully!</strong></p>
        `;
        
        loadingStatus.style.display = 'none';
        modelLoaded = true;
        document.getElementById('toggleRotation').disabled = false;
        
        render();
        
    } catch (error) {
        loadingStatus.textContent = `Error loading model: ${error.message}`;
        loadingStatus.style.color = '#ff0000';
        console.error('Error loading model:', error);
    }
}

// Render function
function render() {
    if (!modelLoaded || !device) return;
    
    updateMatrices();
    
    const canvas = document.getElementById('canvas');
    
    // Handle High DPI
    const devicePixelRatio = window.devicePixelRatio || 1;
    const presentationWidth = canvas.clientWidth * devicePixelRatio;
    const presentationHeight = canvas.clientHeight * devicePixelRatio;

    if (canvas.width !== presentationWidth || canvas.height !== presentationHeight) {
        canvas.width = presentationWidth;
        canvas.height = presentationHeight;
        
        if (depthTexture) depthTexture.destroy();
        if (msaaTexture) msaaTexture.destroy();
        depthTexture = null;
        msaaTexture = null;
    }

    if (!depthTexture) {
        depthTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
            sampleCount: 4
        });
    }

    if (!msaaTexture) {
        msaaTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: navigator.gpu.getPreferredCanvasFormat(),
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
            sampleCount: 4
        });
    }
    
    const commandEncoder = device.createCommandEncoder();
    
    const renderPassDescriptor = {
        colorAttachments: [{
            view: msaaTexture.createView(),
            resolveTarget: context.getCurrentTexture().createView(),
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
            loadOp: 'clear',
            storeOp: 'discard'
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'discard'
        }
    };
    
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(pipeline);
    
    // Draw Body
    if (numBodyIndices > 0) {
        passEncoder.setBindGroup(0, uniformBindGroupBody);
        passEncoder.setVertexBuffer(0, bodyVertexBuffer);
        passEncoder.setVertexBuffer(1, bodyNormalBuffer);
        passEncoder.setVertexBuffer(2, bodyColorBuffer);
        passEncoder.setIndexBuffer(bodyIndexBuffer, 'uint32');
        passEncoder.drawIndexed(numBodyIndices);
    }
    
    // Draw Aileron
    if (numAileronIndices > 0) {
        passEncoder.setBindGroup(0, uniformBindGroupAileron);
        passEncoder.setVertexBuffer(0, aileronVertexBuffer);
        passEncoder.setVertexBuffer(1, aileronNormalBuffer);
        passEncoder.setVertexBuffer(2, aileronColorBuffer);
        passEncoder.setIndexBuffer(aileronIndexBuffer, 'uint32');
        passEncoder.drawIndexed(numAileronIndices);
    }

    // Draw Right Aileron
    if (numRAileronIndices > 0) {
        passEncoder.setBindGroup(0, uniformBindGroupRAileron);
        passEncoder.setVertexBuffer(0, rAileronVertexBuffer);
        passEncoder.setVertexBuffer(1, rAileronNormalBuffer);
        passEncoder.setVertexBuffer(2, rAileronColorBuffer);
        passEncoder.setIndexBuffer(rAileronIndexBuffer, 'uint32');
        passEncoder.drawIndexed(numRAileronIndices);
    }

    // Draw Left Elevator
    if (numLElevatorIndices > 0) {
        passEncoder.setBindGroup(0, uniformBindGroupLElevator);
        passEncoder.setVertexBuffer(0, lElevatorVertexBuffer);
        passEncoder.setVertexBuffer(1, lElevatorNormalBuffer);
        passEncoder.setVertexBuffer(2, lElevatorColorBuffer);
        passEncoder.setIndexBuffer(lElevatorIndexBuffer, 'uint32');
        passEncoder.drawIndexed(numLElevatorIndices);
    }

    // Draw Right Elevator
    if (numRElevatorIndices > 0) {
        passEncoder.setBindGroup(0, uniformBindGroupRElevator);
        passEncoder.setVertexBuffer(0, rElevatorVertexBuffer);
        passEncoder.setVertexBuffer(1, rElevatorNormalBuffer);
        passEncoder.setVertexBuffer(2, rElevatorColorBuffer);
        passEncoder.setIndexBuffer(rElevatorIndexBuffer, 'uint32');
        passEncoder.drawIndexed(numRElevatorIndices);
    }

    // Draw Rudder
    if (numRudderIndices > 0) {
        passEncoder.setBindGroup(0, uniformBindGroupRudder);
        passEncoder.setVertexBuffer(0, rudderVertexBuffer);
        passEncoder.setVertexBuffer(1, rudderNormalBuffer);
        passEncoder.setVertexBuffer(2, rudderColorBuffer);
        passEncoder.setIndexBuffer(rudderIndexBuffer, 'uint32');
        passEncoder.drawIndexed(numRudderIndices);
    }
    
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(render);
}

// Reset view
function resetView() {
    autoRotate = false;
    document.getElementById('toggleRotation').textContent = 'Start Rotation';
    rotationOffset = Math.PI;
    eye = vec3(0, 5, -20);
    at = vec3(0, 0, 0);
    updateMatrices();
}

// Event listeners
document.getElementById('toggleRotation').addEventListener('click', () => {
    autoRotate = !autoRotate;
    const button = document.getElementById('toggleRotation');
    button.textContent = autoRotate ? 'Stop Rotation' : 'Start Rotation';
    
    if (autoRotate) {
        rotationStartTime = Date.now();
    } else {
        const elapsed = (Date.now() - rotationStartTime) * 0.001;
        rotationOffset += elapsed * 0.5;
    }
});

document.getElementById('resetView').addEventListener('click', resetView);

// Keyboard controls
document.addEventListener('keydown', (event) => {
    const step = 5;
    switch(event.key.toLowerCase()) {
        // Roll (A/D)
        case 'a': // Left Aileron Down, Right Aileron Up
            aileronAngle = Math.max(aileronAngle - step, -AILERON_MAX_ANGLE);
            rAileronAngle = Math.max(rAileronAngle - step, -AILERON_MAX_ANGLE);
            console.log('Roll Left (A):', aileronAngle, rAileronAngle);
            break;
        case 'd': // Left Aileron Up, Right Aileron Down
            aileronAngle = Math.min(aileronAngle + step, AILERON_MAX_ANGLE);
            rAileronAngle = Math.min(rAileronAngle + step, AILERON_MAX_ANGLE);
            console.log('Roll Right (D):', aileronAngle, rAileronAngle);
            break;

        // Pitch (W/S)
        case 'w': // Pitch Down
            elevatorAngle = Math.max(elevatorAngle - step, -AILERON_MAX_ANGLE);
            console.log('Pitch Down (W):', elevatorAngle);
            break;
        case 's': // Pitch Up
            elevatorAngle = Math.min(elevatorAngle + step, AILERON_MAX_ANGLE);
            console.log('Pitch Up (S):', elevatorAngle);
            break;

        // Yaw (Q/E)
        case 'q': // Yaw Left
            rudderAngle = Math.max(rudderAngle - step, -AILERON_MAX_ANGLE);
            console.log('Yaw Left (Q):', rudderAngle);
            break;
        case 'e': // Yaw Right
            rudderAngle = Math.min(rudderAngle + step, AILERON_MAX_ANGLE);
            console.log('Yaw Right (E):', rudderAngle);
            break;
    }
});

// Auto-load model on startup
window.addEventListener('load', async () => {
    console.log('Auto-loading model...');
    if (!device) {
        const success = await initWebGPU();
        if (!success) return;
    }
    await loadModel();
});

console.log('Boeing 747 WebGPU Viewer initialized.');
