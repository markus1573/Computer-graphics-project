// Boeing 747 Model Viewer
// WebGL application to load and display a 3D Boeing 747 model

let gl;
let program;
let canvas;

// Model data
let modelData = null;
let vertices = [];
let indices = [];
let normals = [];

// Buffers
let vertexBuffer;
let indexBuffer;
let normalBuffer;

// Matrices and transformations
let modelMatrix;
let viewMatrix;
let projectionMatrix;
let mvpMatrix;

// Animation
let rotation = 0;
let isRotating = false;

// Camera
let cameraDistance = 10.0;
let cameraAngleX = 0;
let cameraAngleY = 0;

// Shader sources
const vertexShaderSource = `
    attribute vec4 a_position;
    attribute vec3 a_normal;
    
    uniform mat4 u_mvpMatrix;
    uniform mat4 u_modelMatrix;
    uniform mat3 u_normalMatrix;
    
    varying vec3 v_normal;
    varying vec3 v_position;
    
    void main() {
        gl_Position = u_mvpMatrix * a_position;
        v_normal = u_normalMatrix * a_normal;
        v_position = (u_modelMatrix * a_position).xyz;
    }
`;

const fragmentShaderSource = `
    precision mediump float;
    
    varying vec3 v_normal;
    varying vec3 v_position;
    
    uniform vec3 u_lightPosition;
    uniform vec3 u_lightColor;
    uniform vec3 u_ambient;
    uniform vec3 u_diffuse;
    uniform vec3 u_specular;
    uniform float u_shininess;
    uniform vec3 u_viewPosition;
    
    void main() {
        vec3 normal = normalize(v_normal);
        vec3 lightDirection = normalize(u_lightPosition - v_position);
        vec3 viewDirection = normalize(u_viewPosition - v_position);
        vec3 reflectDirection = reflect(-lightDirection, normal);
        
        // Ambient
        vec3 ambient = u_ambient * u_lightColor;
        
        // Diffuse
        float diff = max(dot(normal, lightDirection), 0.0);
        vec3 diffuse = diff * u_diffuse * u_lightColor;
        
        // Specular
        float spec = pow(max(dot(viewDirection, reflectDirection), 0.0), u_shininess);
        vec3 specular = spec * u_specular * u_lightColor;
        
        vec3 result = ambient + diffuse + specular;
        gl_FragColor = vec4(result, 1.0);
    }
`;

function initWebGL() {
    canvas = document.getElementById('canvas');
    gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    
    if (!gl) {
        alert('WebGL not supported');
        return false;
    }
    
    // Set viewport
    gl.viewport(0, 0, canvas.width, canvas.height);
    
    // Enable depth testing
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    
    // Set clear color
    gl.clearColor(0.1, 0.1, 0.2, 1.0);
    
    return true;
}

function createShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Error compiling shader:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    
    return shader;
}

function initShaders() {
    const vertexShader = createShader(gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl.FRAGMENT_SHADER, fragmentShaderSource);
    
    if (!vertexShader || !fragmentShader) {
        return false;
    }
    
    program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Error linking program:', gl.getProgramInfoLog(program));
        return false;
    }
    
    gl.useProgram(program);
    
    // Get attribute and uniform locations
    program.positionLocation = gl.getAttribLocation(program, 'a_position');
    program.normalLocation = gl.getAttribLocation(program, 'a_normal');
    
    program.mvpMatrixLocation = gl.getUniformLocation(program, 'u_mvpMatrix');
    program.modelMatrixLocation = gl.getUniformLocation(program, 'u_modelMatrix');
    program.normalMatrixLocation = gl.getUniformLocation(program, 'u_normalMatrix');
    program.lightPositionLocation = gl.getUniformLocation(program, 'u_lightPosition');
    program.lightColorLocation = gl.getUniformLocation(program, 'u_lightColor');
    program.ambientLocation = gl.getUniformLocation(program, 'u_ambient');
    program.diffuseLocation = gl.getUniformLocation(program, 'u_diffuse');
    program.specularLocation = gl.getUniformLocation(program, 'u_specular');
    program.shininessLocation = gl.getUniformLocation(program, 'u_shininess');
    program.viewPositionLocation = gl.getUniformLocation(program, 'u_viewPosition');
    
    return true;
}

function setupBuffers() {
    if (!modelData || vertices.length === 0) {
        console.error('No model data available');
        return false;
    }
    
    // Create and bind vertex buffer
    vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    
    // Create and bind normal buffer
    normalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
    
    // Create and bind index buffer
    indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    
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
    if (!modelData || vertices.length === 0) {
        return;
    }
    
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    
    setupMatrices();
    
    // Set uniforms
    gl.uniformMatrix4fv(program.mvpMatrixLocation, false, flatten(mvpMatrix));
    gl.uniformMatrix4fv(program.modelMatrixLocation, false, flatten(modelMatrix));
    
    // Calculate normal matrix (transpose of inverse of model matrix)
    const normalMatrix = normalMatrixFromMat4(modelMatrix);
    gl.uniformMatrix3fv(program.normalMatrixLocation, false, flatten(normalMatrix));
    
    // Set lighting uniforms
    gl.uniform3fv(program.lightPositionLocation, [10, 10, 10]);
    gl.uniform3fv(program.lightColorLocation, [1, 1, 1]);
    gl.uniform3fv(program.ambientLocation, [0.2, 0.2, 0.3]);
    gl.uniform3fv(program.diffuseLocation, [0.8, 0.8, 0.9]);
    gl.uniform3fv(program.specularLocation, [1, 1, 1]);
    gl.uniform1f(program.shininessLocation, 32.0);
    
    const eye = vec3(
        cameraDistance * Math.sin(cameraAngleY) * Math.cos(cameraAngleX),
        cameraDistance * Math.sin(cameraAngleX),
        cameraDistance * Math.cos(cameraAngleY) * Math.cos(cameraAngleX)
    );
    gl.uniform3fv(program.viewPositionLocation, eye);
    
    // Bind and set up vertex attributes
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.enableVertexAttribArray(program.positionLocation);
    gl.vertexAttribPointer(program.positionLocation, 3, gl.FLOAT, false, 0, 0);
    
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.enableVertexAttribArray(program.normalLocation);
    gl.vertexAttribPointer(program.normalLocation, 3, gl.FLOAT, false, 0, 0);
    
    // Draw
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0);
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
        modelData = await readOBJFile('Boeing 747 with parts.obj', 1.0, false);
        
        if (!modelData) {
            throw new Error('Failed to load OBJ file');
        }
        
        console.log('Model loaded successfully:', modelData);
        
        // Extract vertices, normals, and indices from the model data
        vertices = [];
        normals = [];
        indices = [];
        
        if (modelData.vertices && modelData.normals && modelData.indices) {
            vertices = modelData.vertices;
            normals = modelData.normals;
            indices = modelData.indices;
        } else {
            // Handle different data structure
            console.log('Model data structure:', Object.keys(modelData));
            // Try to extract data from the model structure
            if (modelData.positions) vertices = modelData.positions;
            if (modelData.normals) normals = modelData.normals;
            if (modelData.indices) indices = modelData.indices;
        }
        
        console.log(`Loaded: ${vertices.length/3} vertices, ${normals.length/3} normals, ${indices.length} indices`);
        
        if (vertices.length === 0) {
            throw new Error('No vertex data found in model');
        }
        
        // Set up buffers with the loaded data
        setupBuffers();
        
        // Update UI
        modelInfo.innerHTML = `
            <strong>Model loaded successfully!</strong><br>
            Vertices: ${Math.floor(vertices.length/3)}<br>
            Triangles: ${Math.floor(indices.length/3)}<br>
            Parts: ${modelData.objects ? modelData.objects.length : 'Unknown'}
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

// Initialize the application
function init() {
    if (!initWebGL()) {
        return;
    }
    
    if (!initShaders()) {
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
    
    console.log('Boeing 747 Model Viewer initialized');
}

// Start the application when the page loads
window.addEventListener('load', init);
