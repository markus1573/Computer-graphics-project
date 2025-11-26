// WebGPU Boeing 747 Model Viewer
// Uses OBJParser.js and MV.js for model loading and matrix operations

// Global variables
let device, context, pipeline, uniformBuffer, uniformBindGroup;
let vertexBuffer, normalBuffer, colorBuffer, indexBuffer;
let numIndices = 0;
let modelViewMatrix, projectionMatrix;
let rotationAngle = 0;
let autoRotate = false;
let modelLoaded = false;

// Camera parameters
let eye = vec3(3, 5, 20);
let at = vec3(0, 0, 0);
let up = vec3(0, 1, 0);

// WGSL Shader code
const shaderCode = `
struct Uniforms {
    modelViewMatrix: mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,
    normalMatrix: mat4x4<f32>,
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
    // Use flat shading - calculate normal from screen-space derivatives
    // This ignores the potentially bad normals from the model
    let dpdx = dpdx(input.viewPosition);
    let dpdy = dpdy(input.viewPosition);
    let faceNormal = normalize(cross(dpdx, dpdy));
    
    // Three-point lighting for better depth and shape
    // Key light: Main light from front-top-right
    let keyLightDir = normalize(vec3<f32>(0.5, 0.5, 1.0));
    let keyIntensity = 0.6;
    
    // Fill light: Softer light from front-left to fill shadows
    let fillLightDir = normalize(vec3<f32>(-0.3, 0.2, 0.8));
    let fillIntensity = 0.3;
    
    // Rim light: From behind to highlight edges
    let rimLightDir = normalize(vec3<f32>(0.0, 0.3, -1.0));
    let rimIntensity = 0.2;
    
    // Ambient light for base visibility
    let ambient = 0.4;
    
    // Calculate lighting contributions
    let keyDiffuse = keyIntensity * max(dot(faceNormal, keyLightDir), 0.0);
    let fillDiffuse = fillIntensity * max(dot(faceNormal, fillLightDir), 0.0);
    let rimDiffuse = rimIntensity * max(dot(faceNormal, rimLightDir), 0.0);
    
    // Combine all lighting
    let lighting = ambient + keyDiffuse + fillDiffuse + rimDiffuse;
    let finalColor = input.color.rgb * lighting;
    
    return vec4<f32>(finalColor, 1.0);
}
`;

// Initialize WebGPU
async function initWebGPU() {
    const canvas = document.getElementById('canvas');
    
    // Check for WebGPU support
    if (!navigator.gpu) {
        alert('WebGPU is not supported in your browser. Please use Chrome or Edge with WebGPU enabled.');
        return false;
    }
    
    // Request adapter and device
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        alert('Failed to get GPU adapter');
        return false;
    }
    
    device = await adapter.requestDevice();
    
    // Configure canvas context
    context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
        alphaMode: 'opaque',
    });
    
    // Create shader module
    const shaderModule = device.createShaderModule({
        code: shaderCode
    });
    
    // Create pipeline
    pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
            buffers: [
                {
                    arrayStride: 16, // 4 floats * 4 bytes
                    attributes: [{
                        shaderLocation: 0,
                        offset: 0,
                        format: 'float32x4'
                    }]
                },
                {
                    arrayStride: 16,
                    attributes: [{
                        shaderLocation: 1,
                        offset: 0,
                        format: 'float32x4'
                    }]
                },
                {
                    arrayStride: 16,
                    attributes: [{
                        shaderLocation: 2,
                        offset: 0,
                        format: 'float32x4'
                    }]
                }
            ]
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{
                format: canvasFormat
            }]
        },
        primitive: {
            topology: 'triangle-list',
            cullMode: 'back',
            frontFace: 'ccw'
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus'
        }
    });
    
    // Create uniform buffer
    const uniformBufferSize = 3 * 16 * 4; // 3 mat4x4 (each 16 floats * 4 bytes)
    uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    // Create bind group
    uniformBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: {
                buffer: uniformBuffer
            }
        }]
    });
    
    // Initialize matrices
    updateMatrices();
    
    return true;
}

// Update transformation matrices
function updateMatrices() {
    // Model transformation - translate to origin, rotate, and scale
    const transMatrix = translate(0, 2.7, 1.5); // Move model center to origin
    const rotMatrix = rotateY(rotationAngle);
    const scaleMatrix = scalem(2.0, 2.0, 2.0); // Scale up for visibility
    
    // Apply transformations: Scale -> Rotate -> Translate
    let modelMatrix = scaleMatrix;
    modelMatrix = mult(rotMatrix, modelMatrix);
    modelMatrix = mult(transMatrix, modelMatrix);
    
    // View matrix
    const viewMatrix = lookAt(eye, at, up);
    
    // Model-View matrix
    modelViewMatrix = mult(viewMatrix, modelMatrix);
    
    // Projection matrix
    const canvas = document.getElementById('canvas');
    const aspect = canvas.width / canvas.height;
    projectionMatrix = perspective(45, aspect, 0.1, 100.0);
    
    // Normal matrix (inverse transpose of model-view)
    const normalMat = normalMatrix(modelViewMatrix, true);
    
    // Update uniform buffer
    if (device && uniformBuffer) {
        // MV.js flatten() already converts row-major to column-major for WebGPU
        const matrixData = new Float32Array([
            ...flatten(modelViewMatrix),
            ...flatten(projectionMatrix),
            ...flatten(normalMat)
        ]);
        device.queue.writeBuffer(uniformBuffer, 0, matrixData);
        
        // Debug: log matrices once
        if (!updateMatrices.logged) {
            console.log('Model matrix:', modelMatrix);
            console.log('View matrix:', viewMatrix);
            console.log('Projection matrix:', projectionMatrix);
            console.log('Eye:', eye, 'At:', at);
            console.log('Matrix data length:', matrixData.length);
            console.log('First 16 floats (modelView):', matrixData.slice(0, 16));
            updateMatrices.logged = true;
        }
    }
}

// Load and parse OBJ file
async function loadModel() {
    const loadingStatus = document.getElementById('loadingStatus');
    const modelInfo = document.getElementById('modelInfo');
    loadingStatus.style.display = 'block';
    loadingStatus.textContent = 'Loading Boeing 747 model...';
    
    try {
        // Load the OBJ file using OBJParser
        const drawingInfo = await readOBJFile('Boeing747_with_parts.obj', 1.0, false);
        
        if (!drawingInfo) {
            throw new Error('Failed to load OBJ file');
        }
        
        // Display model information
        numIndices = drawingInfo.indices.length;
        console.log('Model loaded:', {
            vertices: drawingInfo.vertices.length / 4,
            indices: numIndices,
            triangles: numIndices / 3
        });
        console.log('First few vertices:', drawingInfo.vertices.slice(0, 12));
        modelInfo.innerHTML = `
            <p><strong>Model loaded successfully!</strong></p>
            <p>Vertices: ${drawingInfo.vertices.length / 4}</p>
            <p>Triangles: ${numIndices / 3}</p>
            <p>Normals: ${drawingInfo.normals.length / 4}</p>
        `;
        
        // Create vertex buffer
        vertexBuffer = device.createBuffer({
            size: drawingInfo.vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(vertexBuffer, 0, drawingInfo.vertices);
        
        // Create normal buffer
        normalBuffer = device.createBuffer({
            size: drawingInfo.normals.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(normalBuffer, 0, drawingInfo.normals);
        
        // Create color buffer
        colorBuffer = device.createBuffer({
            size: drawingInfo.colors.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(colorBuffer, 0, drawingInfo.colors);
        
        // Create index buffer
        indexBuffer = device.createBuffer({
            size: drawingInfo.indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(indexBuffer, 0, drawingInfo.indices);
        
        loadingStatus.style.display = 'none';
        modelLoaded = true;
        
        // Enable controls
        document.getElementById('toggleRotation').disabled = false;
        
        // Start rendering
        render();
        
    } catch (error) {
        loadingStatus.textContent = `Error loading model: ${error.message}`;
        loadingStatus.style.color = '#ff0000';
        console.error('Error loading model:', error);
    }
}

// Render function
function render() {
    if (!modelLoaded || !device) {
        console.log('Render skipped - modelLoaded:', modelLoaded, 'device:', !!device);
        return;
    }
    
    // Update rotation if auto-rotate is enabled
    if (autoRotate) {
        rotationAngle += 0.5;
        updateMatrices();
    }
    
    // Create depth texture
    const canvas = document.getElementById('canvas');
    const depthTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    
    // Create command encoder
    const commandEncoder = device.createCommandEncoder();
    
    // Create render pass
    const renderPassDescriptor = {
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0.2, g: 0.3, b: 0.4, a: 1.0 }, // Brighter blue so we know rendering works
            loadOp: 'clear',
            storeOp: 'store'
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store'
        }
    };
    
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, uniformBindGroup);
    passEncoder.setVertexBuffer(0, vertexBuffer);
    passEncoder.setVertexBuffer(1, normalBuffer);
    passEncoder.setVertexBuffer(2, colorBuffer);
    passEncoder.setIndexBuffer(indexBuffer, 'uint32');
    passEncoder.drawIndexed(numIndices);
    passEncoder.end();
    
    // Debug: log first render
    if (!render.logged) {
        console.log('Rendering with', numIndices, 'indices');
        console.log('Buffers:', !!vertexBuffer, !!normalBuffer, !!colorBuffer, !!indexBuffer);
        render.logged = true;
    }
    
    // Submit commands
    device.queue.submit([commandEncoder.finish()]);
    
    // Request next frame
    requestAnimationFrame(render);
}

// Reset view
function resetView() {
    rotationAngle = 0;
    eye = vec3(0, 0, 20);
    at = vec3(0, 0, 0);
    updateMatrices();
}

// Event listeners
document.getElementById('loadModel').addEventListener('click', async () => {
    if (!device) {
        const success = await initWebGPU();
        if (!success) return;
    }
    await loadModel();
});

document.getElementById('toggleRotation').addEventListener('click', () => {
    autoRotate = !autoRotate;
    const button = document.getElementById('toggleRotation');
    button.textContent = autoRotate ? 'Stop Rotation' : 'Start Rotation';
});

document.getElementById('resetView').addEventListener('click', resetView);

// Aileron controls (for future implementation)
document.getElementById('aileronUp').addEventListener('click', () => {
    console.log('Aileron up - control parts animation to be implemented');
});

document.getElementById('aileronDown').addEventListener('click', () => {
    console.log('Aileron down - control parts animation to be implemented');
});

document.getElementById('aileronReset').addEventListener('click', () => {
    console.log('Aileron reset - control parts animation to be implemented');
});

// Initialize on page load
console.log('Boeing 747 WebGPU Viewer initialized. Click "Load Model" to begin.');
