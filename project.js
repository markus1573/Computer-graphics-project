// Boeing 747 Model Viewer
// WebGPU application to load and display a 3D Boeing 747 model

let device;
let context;
let pipeline;
let canvas;

// Model data
let modelData = null;
let vertices = [];
let indices = [];
let normals = [];
let parts = {}; // Store each part separately

// Matrices and transformations
let modelMatrix;
let viewMatrix;
let projectionMatrix;
let mvpMatrix;

// Animation
let rotation = 0;
let isRotating = false;

// Control surface angles (in degrees)
let leftAileronAngle = 0;

// Camera
let cameraDistance = 10.0;
let cameraAngleX = 0;
let cameraAngleY = 0;

// WebGPU resources
let depthTexture;
let uniformBuffer;
let uniformBindGroup;
let hingeMarkerBuffer;
let hingeMarkerBindGroup;

// Shader sources (WGSL)
const shaderSource = `
struct Uniforms {
    mvpMatrix: mat4x4<f32>,
    modelMatrix: mat4x4<f32>,
    normalMatrix0: vec3<f32>,
    _pad0: f32,
    normalMatrix1: vec3<f32>,
    _pad1: f32,
    normalMatrix2: vec3<f32>,
    _pad2: f32,
    lightPosition: vec3<f32>,
    _pad3: f32,
    viewPosition: vec3<f32>,
    _pad4: f32,
    lightColor: vec3<f32>,
    _pad5: f32,
    ambient: vec3<f32>,
    _pad6: f32,
    diffuse: vec3<f32>,
    _pad7: f32,
    specular: vec3<f32>,
    shininess: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) worldPosition: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let pos4 = vec4<f32>(input.position, 1.0);
    output.position = uniforms.mvpMatrix * pos4;
    output.worldPosition = (uniforms.modelMatrix * pos4).xyz;
    
    // Reconstruct normal matrix from padded rows
    let normalMatrix = mat3x3<f32>(
        uniforms.normalMatrix0,
        uniforms.normalMatrix1,
        uniforms.normalMatrix2
    );
    output.normal = normalMatrix * input.normal;
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    let lightDir = normalize(uniforms.lightPosition - input.worldPosition);
    let viewDir = normalize(uniforms.viewPosition - input.worldPosition);
    let reflectDir = reflect(-lightDir, normal);
    
    // Ambient
    let ambient = uniforms.ambient * uniforms.lightColor;
    
    // Diffuse
    let diff = max(dot(normal, lightDir), 0.0);
    let diffuse = diff * uniforms.diffuse * uniforms.lightColor;
    
    // Specular
    let spec = pow(max(dot(viewDir, reflectDir), 0.0), uniforms.shininess);
    let specular = spec * uniforms.specular * uniforms.lightColor;
    
    let result = ambient + diffuse + specular;
    return vec4<f32>(result, 1.0);
}

// Hinge marker shader
@vertex
fn hingeVertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 1.0);
    output.worldPosition = input.position;
    output.normal = vec3<f32>(0.0, 1.0, 0.0);
    return output;
}

@fragment
fn hingeFragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red color
}
`;

async function initWebGPU() {
    canvas = document.getElementById('canvas');
    
    if (!navigator.gpu) {
        alert('WebGPU not supported in this browser');
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
        alphaMode: 'opaque',
    });
    
    // Create depth texture
    depthTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    
    return true;
}

async function initPipeline() {
    const shaderModule = device.createShaderModule({
        code: shaderSource,
    });
    
    // Create uniform buffer (larger to accommodate all uniforms)
    // mat4x4 (64) + mat4x4 (64) + mat3x3 as 3xvec4 (48) + 7 vec4s (112) = 288, round to 320
    const uniformBufferSize = 320;
    uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    
    const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: 'uniform' }
        }]
    });
    
    uniformBindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }]
    });
    
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
    });
    
    pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
            buffers: [{
                arrayStride: 12, // 3 floats * 4 bytes
                attributes: [{
                    shaderLocation: 0,
                    offset: 0,
                    format: 'float32x3'
                }]
            }, {
                arrayStride: 12, // 3 floats * 4 bytes
                attributes: [{
                    shaderLocation: 1,
                    offset: 0,
                    format: 'float32x3'
                }]
            }]
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{
                format: navigator.gpu.getPreferredCanvasFormat()
            }]
        },
        primitive: {
            topology: 'triangle-list',
            cullMode: 'back',
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus',
        }
    });
    
    return true;
}

function setupBuffers() {
    if (!modelData || vertices.length === 0) {
        console.error('No model data available');
        return false;
    }
    
    // Create buffers for each part
    for (let partName in parts) {
        const part = parts[partName];
        
        // Create vertex buffer
        part.vertexBuffer = device.createBuffer({
            size: part.vertices.length * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(part.vertexBuffer, 0, new Float32Array(part.vertices));
        
        // Create normal buffer
        part.normalBuffer = device.createBuffer({
            size: part.normals.length * 4,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(part.normalBuffer, 0, new Float32Array(part.normals));
        
        // Create index buffer
        part.indexBuffer = device.createBuffer({
            size: part.indices.length * 2, // Uint16
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(part.indexBuffer, 0, new Uint16Array(part.indices));
    }
    
    console.log('Buffers created for all parts');
    return true;
}

function setupMatrices() {
    // Model matrix
    modelMatrix = mat4();
    modelMatrix = mult(modelMatrix, rotateY(rotation));
    
    // View matrix
    const eye = vec3(
        cameraDistance * Math.sin(cameraAngleY) * Math.cos(cameraAngleX),
        cameraDistance * Math.sin(cameraAngleX),
        cameraDistance * Math.cos(cameraAngleY) * Math.cos(cameraAngleX)
    );
    const at = vec3(0, 0, 0);
    const up = vec3(0, 1, 0);
    viewMatrix = lookAt(eye, at, up);
    
    // Projection matrix
    const aspect = canvas.width / canvas.height;
    projectionMatrix = perspective(45, aspect, 0.1, 100);
    
    // Combined MVP matrix
    mvpMatrix = mult(projectionMatrix, mult(viewMatrix, modelMatrix));
}

function render() {
    if (!modelData || vertices.length === 0 || !device || !pipeline) {
        return;
    }
    
    // Check if parts exist
    if (Object.keys(parts).length === 0) {
        console.warn('No parts available to render');
        return;
    }
    
    setupMatrices();
    
    const eye = vec3(
        cameraDistance * Math.sin(cameraAngleY) * Math.cos(cameraAngleX),
        cameraDistance * Math.sin(cameraAngleX),
        cameraDistance * Math.cos(cameraAngleY) * Math.cos(cameraAngleX)
    );
    
    const commandEncoder = device.createCommandEncoder();
    
    const textureView = context.getCurrentTexture().createView();
    
    const renderPassDescriptor = {
        colorAttachments: [{
            view: textureView,
            clearValue: { r: 0.1, g: 0.1, b: 0.2, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        }
    };
    
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(pipeline);
    
    // Draw each part with its own transformation
    let partsDrawn = 0;
    for (let partName in parts) {
        const part = parts[partName];
        partsDrawn++;
        
        // Calculate part-specific transformation
        let partModelMatrix = modelMatrix;
        
        // Apply aileron rotation for left_aileron
        if (partName.includes('left_aileron') || partName.includes('left aileron')) {
            const bounds = part.bounds;
            const xLen = bounds.maxX - bounds.minX;
            const yLen = bounds.maxY - bounds.minY;
            const zLen = bounds.maxZ - bounds.minZ;
            
            let hingeX, hingeY, hingeZ, rotationAxis;
            
            if (!render.loggedAileronHinge) {
                console.log(`Aileron dimensions: X=${xLen.toFixed(2)}, Y=${yLen.toFixed(2)}, Z=${zLen.toFixed(2)}`);
            }
            
            hingeX = bounds.minX;
            hingeY = bounds.minY;
            hingeZ = (bounds.minZ + bounds.maxZ) / 2;
            rotationAxis = 'Z';

            if (!render.loggedAileronHinge) {
                console.log(`Aileron hinge: axis=${rotationAxis}, point=(${hingeX.toFixed(2)}, ${hingeY.toFixed(2)}, ${hingeZ.toFixed(2)})`);
                render.loggedAileronHinge = true;
            }
            
            if (!render.hingePoint) {
                render.hingePoint = { x: hingeX, y: hingeY, z: hingeZ };
            }
            
            partModelMatrix = mult(partModelMatrix, translate(hingeX, hingeY, hingeZ));
            partModelMatrix = mult(partModelMatrix, rotateX(leftAileronAngle));
            partModelMatrix = mult(partModelMatrix, translate(-hingeX, -hingeY, -hingeZ));
        }
        
        // Calculate MVP for this part
        const partMvp = mult(projectionMatrix, mult(viewMatrix, partModelMatrix));
        const normalMatrix = normalMatrixFromMat4(partModelMatrix);
        
        // Update uniforms
        updateUniforms(partMvp, partModelMatrix, normalMatrix, eye);
        
        // Set vertex buffers and draw
        passEncoder.setVertexBuffer(0, part.vertexBuffer);
        passEncoder.setVertexBuffer(1, part.normalBuffer);
        passEncoder.setIndexBuffer(part.indexBuffer, 'uint16');
        passEncoder.setBindGroup(0, uniformBindGroup);
        passEncoder.drawIndexed(part.indices.length);
    }
    
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    
    // Log once per second how many parts were drawn
    if (!render.lastLog || Date.now() - render.lastLog > 1000) {
        console.log(`Drew ${partsDrawn} parts`);
        render.lastLog = Date.now();
    }
}

function updateUniforms(mvpMatrix, modelMatrix, normalMatrix, viewPosition) {
    // Pack uniforms into buffer
    const uniformData = new Float32Array(80); // Increased size
    let offset = 0;
    
    // MVP matrix (16 floats)
    const mvpFlat = flatten(mvpMatrix);
    for (let i = 0; i < mvpFlat.length; i++) {
        uniformData[offset++] = mvpFlat[i];
    }
    
    // Model matrix (16 floats)
    const modelFlat = flatten(modelMatrix);
    for (let i = 0; i < modelFlat.length; i++) {
        uniformData[offset++] = modelFlat[i];
    }
    
    // Normal matrix (9 floats as 3x vec3, each padded to vec4 = 12 floats total)
    // Row 0
    uniformData[offset++] = normalMatrix[0];
    uniformData[offset++] = normalMatrix[1];
    uniformData[offset++] = normalMatrix[2];
    uniformData[offset++] = 0; // padding
    // Row 1
    uniformData[offset++] = normalMatrix[3];
    uniformData[offset++] = normalMatrix[4];
    uniformData[offset++] = normalMatrix[5];
    uniformData[offset++] = 0; // padding
    // Row 2
    uniformData[offset++] = normalMatrix[6];
    uniformData[offset++] = normalMatrix[7];
    uniformData[offset++] = normalMatrix[8];
    uniformData[offset++] = 0; // padding
    
    // Light position (3 floats + 1 padding)
    uniformData[offset++] = 10;
    uniformData[offset++] = 10;
    uniformData[offset++] = 10;
    uniformData[offset++] = 0;
    
    // View position (3 floats + 1 padding)
    uniformData[offset++] = viewPosition[0];
    uniformData[offset++] = viewPosition[1];
    uniformData[offset++] = viewPosition[2];
    uniformData[offset++] = 0;
    
    // Light color (3 floats + 1 padding)
    uniformData[offset++] = 1;
    uniformData[offset++] = 1;
    uniformData[offset++] = 1;
    uniformData[offset++] = 0;
    
    // Ambient (3 floats + 1 padding)
    uniformData[offset++] = 0.2;
    uniformData[offset++] = 0.2;
    uniformData[offset++] = 0.3;
    uniformData[offset++] = 0;
    
    // Diffuse (3 floats + 1 padding)
    uniformData[offset++] = 0.8;
    uniformData[offset++] = 0.8;
    uniformData[offset++] = 0.9;
    uniformData[offset++] = 0;
    
    // Specular (3 floats + 1 padding)
    uniformData[offset++] = 1;
    uniformData[offset++] = 1;
    uniformData[offset++] = 1;
    uniformData[offset++] = 0;
    
    // Shininess (1 float + 3 padding)
    uniformData[offset++] = 32.0;
    uniformData[offset++] = 0;
    uniformData[offset++] = 0;
    uniformData[offset++] = 0;
    
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
}

function animate() {
    if (isRotating) {
        rotation += 0.01;
    }
    render();
    requestAnimationFrame(animate);
}

function normalMatrixFromMat4(m) {
    // Extract 3x3 rotation part and transpose
    return [
        m[0][0], m[1][0], m[2][0],
        m[0][1], m[1][1], m[2][1],
        m[0][2], m[1][2], m[2][2]
    ];
}

function parseOBJParts(objText) {
    const lines = objText.split('\n');
    const parts = {};
    let currentPart = 'default';
    let faceIndex = 0;
    
    for (let line of lines) {
        line = line.trim();
        
        // Check for object/group definitions
        if (line.startsWith('g ') || line.startsWith('o ')) {
            const partName = line.substring(2).trim();
            if (partName) {
                currentPart = partName;
                if (!parts[currentPart]) {
                    parts[currentPart] = { startFace: faceIndex, endFace: faceIndex };
                }
            }
        }
        // Count faces
        else if (line.startsWith('f ')) {
            if (!parts[currentPart]) {
                parts[currentPart] = { startFace: faceIndex, endFace: faceIndex };
            }
            parts[currentPart].endFace = faceIndex + 1;
            faceIndex++;
        }
    }
    
    return parts;
}

async function loadModel() {
    const loadingStatus = document.getElementById('loadingStatus');
    const modelInfo = document.getElementById('modelInfo');
    const loadButton = document.getElementById('loadModel');
    const toggleButton = document.getElementById('toggleRotation');
    
    loadingStatus.style.display = 'block';
    loadButton.disabled = true;
    modelInfo.innerHTML = 'Loading...';
    
    try {
        console.log('Loading Boeing 747 model...');
        
        // Clear previous model data
        parts = {};
        vertices = [];
        normals = [];
        indices = [];
        
        // Load the full model with OBJParser
        // We need to modify readOBJFile to return the OBJDoc before it merges
        const response = await fetch('Boeing747_with_parts.obj');
        const objText = await response.text();
        
        const objDoc = new OBJDoc('Boeing747_with_parts.obj');
        const result = await objDoc.parse(objText, 1.0, false);
        
        if (!result) {
            throw new Error('Failed to parse OBJ file');
        }
        
        console.log('Model parsed successfully');
        console.log('Found', objDoc.objects.length, 'objects');
        
        // Now extract each object separately
        for (let i = 0; i < objDoc.objects.length; i++) {
            const obj = objDoc.objects[i];
            console.log(`Processing object: ${obj.name}, faces: ${obj.faces.length}`);
            
            const partVertices = [];
            const partNormals = [];
            const partIndices = [];
            const vertexMap = new Map(); // Map old vertex index to new
            let newVertexIndex = 0;
            
            // Process each face of this object
            for (let j = 0; j < obj.faces.length; j++) {
                const face = obj.faces[j];
                const faceNormal = face.normal;
                
                // Process each vertex in the face
                for (let k = 0; k < face.vIndices.length; k++) {
                    const vIdx = face.vIndices[k];
                    const nIdx = face.nIndices[k];
                    
                    // Create a unique key for this vertex+normal combination
                    const key = `${vIdx}_${nIdx}`;
                    
                    if (!vertexMap.has(key)) {
                        // Add new vertex
                        const vertex = objDoc.vertices[vIdx];
                        partVertices.push(vertex.x, vertex.y, vertex.z);
                        
                        // Add normal
                        if (nIdx >= 0) {
                            const normal = objDoc.normals[nIdx];
                            partNormals.push(normal.x, normal.y, normal.z);
                        } else {
                            partNormals.push(faceNormal.x, faceNormal.y, faceNormal.z);
                        }
                        
                        vertexMap.set(key, newVertexIndex);
                        newVertexIndex++;
                    }
                    
                    // Add index
                    partIndices.push(vertexMap.get(key));
                }
            }
            
            // Store this part
            if (partVertices.length > 0) {
                // Calculate bounding box for this part
                let minX = Infinity, minY = Infinity, minZ = Infinity;
                let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
                
                for (let v = 0; v < partVertices.length; v += 3) {
                    minX = Math.min(minX, partVertices[v]);
                    maxX = Math.max(maxX, partVertices[v]);
                    minY = Math.min(minY, partVertices[v + 1]);
                    maxY = Math.max(maxY, partVertices[v + 1]);
                    minZ = Math.min(minZ, partVertices[v + 2]);
                    maxZ = Math.max(maxZ, partVertices[v + 2]);
                }
                
                parts[obj.name] = {
                    vertices: partVertices,
                    normals: partNormals,
                    indices: partIndices,
                    vertexBuffer: null,
                    normalBuffer: null,
                    indexBuffer: null,
                    bounds: { minX, maxX, minY, maxY, minZ, maxZ }
                };
                
                console.log(`✓ Part ${obj.name}: ${partVertices.length/3} vertices, ${partIndices.length} indices`);
                if (obj.name.includes('aileron')) {
                    console.log(`  Bounds: X[${minX.toFixed(2)}, ${maxX.toFixed(2)}], Y[${minY.toFixed(2)}, ${maxY.toFixed(2)}], Z[${minZ.toFixed(2)}, ${maxZ.toFixed(2)}]`);
                }
            }
        }
        
        // Get the merged DrawingInfo for backward compatibility
        modelData = objDoc.getDrawingInfo();
        
        // Extract vertices and normals from merged data
        if (modelData.vertices && modelData.normals && modelData.indices) {
            // Convert 4-component vertices to 3-component (skip w component)
            for (let i = 0; i < modelData.vertices.length; i += 4) {
                vertices.push(modelData.vertices[i]);     // x
                vertices.push(modelData.vertices[i + 1]); // y
                vertices.push(modelData.vertices[i + 2]); // z
            }
            
            // Convert 4-component normals to 3-component (skip w component)
            for (let i = 0; i < modelData.normals.length; i += 4) {
                normals.push(modelData.normals[i]);     // x
                normals.push(modelData.normals[i + 1]); // y
                normals.push(modelData.normals[i + 2]); // z
            }
            
            // Copy indices as-is
            indices = Array.from(modelData.indices);
        } else {
            console.error('Unexpected model data structure:', Object.keys(modelData));
        }
        
        console.log(`Loaded: ${vertices.length/3} vertices, ${normals.length/3} normals, ${indices.length} indices`);
        console.log(`Total parts: ${Object.keys(parts).length}`);
        console.log(`Parts list:`, Object.keys(parts));
        console.log('=== PART NAMES (for control surfaces) ===');
        Object.keys(parts).forEach(name => {
            console.log(`  - "${name}"`);
        });
        console.log('=========================================');
        
        if (vertices.length === 0) {
            throw new Error('No vertex data found in model');
        }
        
        // Set up buffers with the loaded data
        setupBuffers();
        
        // Update UI
        const numParts = modelData.objects ? modelData.objects.length : 0;
        const partNames = modelData.objects ? modelData.objects.map(obj => obj.name).join(', ') : 'N/A';
        modelInfo.innerHTML = `
            <strong>Model loaded successfully!</strong><br>
            Vertices: ${Math.floor(vertices.length/3)}<br>
            Triangles: ${Math.floor(indices.length/3)}<br>
            Parts: ${numParts}<br>
            <small>Parts: ${partNames}</small>
        `;
        
        toggleButton.disabled = false;
        isRotating = true;
        
    } catch (error) {
        console.error('Error loading model:', error);
        modelInfo.innerHTML = `<span style="color: #ff6b6b;">Error loading model: ${error.message}</span>`;
    } finally {
        loadingStatus.style.display = 'none';
        loadButton.disabled = false;
    }
}

function resetView() {
    rotation = 0;
    cameraDistance = 10.0;
    cameraAngleX = 0;
    cameraAngleY = 0;
}

function aileronUp() {
    leftAileronAngle = Math.min(leftAileronAngle + 5, 30); // Max 30 degrees up
    console.log('Left aileron angle:', leftAileronAngle);
}

function aileronDown() {
    leftAileronAngle = Math.max(leftAileronAngle - 5, -30); // Max 30 degrees down
    console.log('Left aileron angle:', leftAileronAngle);
}

function aileronReset() {
    leftAileronAngle = 0;
    console.log('Left aileron reset to:', leftAileronAngle);
}

// Initialize the application
async function init() {
    if (!await initWebGPU()) {
        return;
    }
    
    if (!await initPipeline()) {
        return;
    }
    
    // Set up event listeners
    document.getElementById('loadModel').addEventListener('click', loadModel);
    document.getElementById('toggleRotation').addEventListener('click', () => {
        isRotating = !isRotating;
        document.getElementById('toggleRotation').textContent = 
            isRotating ? 'Stop Rotation' : 'Start Rotation';
    });
    document.getElementById('resetView').addEventListener('click', resetView);
    
    // Add aileron control event listeners
    document.getElementById('aileronUp').addEventListener('click', aileronUp);
    document.getElementById('aileronDown').addEventListener('click', aileronDown);
    document.getElementById('aileronReset').addEventListener('click', aileronReset);
    
    // Add mouse controls for camera
    let mouseDown = false;
    let lastMouseX = 0;
    let lastMouseY = 0;
    
    canvas.addEventListener('mousedown', (e) => {
        mouseDown = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
    });
    
    canvas.addEventListener('mousemove', (e) => {
        if (!mouseDown) return;
        
        const deltaX = e.clientX - lastMouseX;
        const deltaY = e.clientY - lastMouseY;
        
        cameraAngleY += deltaX * 0.01;
        cameraAngleX += deltaY * 0.01;
        
        // Clamp vertical angle
        cameraAngleX = Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1, cameraAngleX));
        
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
    });
    
    canvas.addEventListener('mouseup', () => {
        mouseDown = false;
    });
    
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        cameraDistance += e.deltaY * 0.01;
        cameraDistance = Math.max(2, Math.min(50, cameraDistance));
    });
    
    // Start the render loop
    animate();
    
    console.log('Boeing 747 Model Viewer initialized (WebGPU)');
}

// Start the application when the page loads
window.addEventListener('load', init);
