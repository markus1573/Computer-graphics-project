// WebGPU Triangle Mesh Renderer - Part 4
// Phong Lighting and Shading

// Matrix functions are now provided by MV.js

class WebGPURenderer {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.device = null;
        this.context = null;
        this.renderPipeline = null;
        this.wireframePipeline = null; // Separate pipeline for wireframe overlay
        this.vertexBuffer = null;
        this.normalBuffer = null;
        this.barycoordBuffer = null; // For wireframe rendering
        this.indexBuffer = null;
        this.uniformBuffer = null;
        this.modelData = null;
        this.isWireframe = false; // Track wireframe mode
        this.isRotating = true; // Track rotation state
        this.rotationStartTime = Date.now(); // When rotation started
        this.rotationOffset = 0; // Offset to maintain continuity
        
        // Lighting parameters
        this.lightingParams = {
            kd: 0.8,        // Diffuse coefficient
            ks: 0.5,        // Specular coefficient
            shininess: 32,  // Phong exponent
            le: 1.0,        // Light emission
            la: 0.1         // Ambient light
        };
        
        this.init();
    }

    async init() {
        try {
            // Check WebGPU support
            if (!navigator.gpu) {
                throw new Error('WebGPU not supported');
            }

            // Request adapter and device
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                throw new Error('No WebGPU adapter found');
            }

            this.device = await adapter.requestDevice();
            this.context = this.canvas.getContext('webgpu');
            
            // Set up high-DPI canvas
            this.setupHighDPICanvas();
            
            // Configure canvas
            const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
            this.context.configure({
                device: this.device,
                format: canvasFormat,
            });

            console.log('WebGPU initialized successfully');
            this.setupEventListeners();
            this.setupSliders();
            
        } catch (error) {
            console.error('Failed to initialize WebGPU:', error);
            document.getElementById('info').innerHTML = 
                '<p style="color: red;">WebGPU initialization failed: ' + error.message + '</p>';
        }
    }

    setupHighDPICanvas() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        
        // Set the actual size in memory (scaled to account for extra pixel density)
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        
        // Scale the canvas back down using CSS
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        
        console.log('High-DPI canvas setup:', {
            devicePixelRatio: dpr,
            canvasSize: `${this.canvas.width}x${this.canvas.height}`,
            displaySize: `${rect.width}x${rect.height}`
        });
    }

    setupEventListeners() {
        document.getElementById('loadModel').addEventListener('click', () => {
            this.loadModel(false); // Standard rendering
        });
        
        document.getElementById('loadWireframe').addEventListener('click', () => {
            this.loadModel(true); // Wireframe rendering
        });
        
        document.getElementById('toggleRotation').addEventListener('click', () => {
            this.toggleRotation();
        });
    }

    setupSliders() {
        // Setup lighting parameter sliders
        const sliders = [
            { id: 'kd-slider', valueId: 'kd-value', param: 'kd' },
            { id: 'ks-slider', valueId: 'ks-value', param: 'ks' },
            { id: 'shininess-slider', valueId: 'shininess-value', param: 'shininess' },
            { id: 'le-slider', valueId: 'le-value', param: 'le' },
            { id: 'la-slider', valueId: 'la-value', param: 'la' }
        ];

        sliders.forEach(({ id, valueId, param }) => {
            const slider = document.getElementById(id);
            const valueDisplay = document.getElementById(valueId);
            
            slider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                this.lightingParams[param] = value;
                valueDisplay.textContent = value.toFixed(2);
                
                // Re-render if model is loaded
                if (this.modelData) {
                    this.render();
                }
            });
            
            // Initialize display
            valueDisplay.textContent = this.lightingParams[param].toFixed(2);
        });
    }

    toggleRotation() {
        this.isRotating = !this.isRotating;
        const button = document.getElementById('toggleRotation');
        button.textContent = this.isRotating ? 'Stop Rotation' : 'Start Rotation';
        console.log('Rotation toggled:', this.isRotating ? 'ON' : 'OFF');
        
        if (this.isRotating) {
            // Starting rotation - set new start time
            this.rotationStartTime = Date.now();
        } else {
            // Stopping rotation - calculate offset to maintain current position
            const elapsed = (Date.now() - this.rotationStartTime) * 0.001;
            this.rotationOffset += elapsed * 0.5;
        }
    }

    async loadModel(isWireframe = false) {
        try {
            document.getElementById('info').innerHTML = '<p>Loading model...</p>';
            
            // Load OBJ file using appropriate parser
            const modelData = isWireframe ? 
                await readOBJFileWireframe('../monke.obj', 1.0, false) : // Wireframe parser
                await readOBJFileStandard('../monke.obj', 1.0, false);   // Standard parser
            
            if (!modelData) {
                throw new Error('Failed to load OBJ file');
            }

            this.modelData = modelData;
            this.isWireframe = isWireframe; // Store wireframe mode
            console.log('Model loaded:', modelData);
            console.log('Vertex count:', modelData.vertices.length / 4);
            console.log('Index count:', modelData.indices.length);
            console.log('Wireframe mode:', isWireframe);
            
            // Create buffers and render pipeline
            this.createBuffers();
            this.createRenderPipeline();
            this.createWireframePipeline(); // Create wireframe pipeline for overlay
            this.render();
            
            const renderMode = isWireframe ? 'wireframe' : 'standard';
            document.getElementById('info').innerHTML = 
                `<p>Model loaded successfully in ${renderMode} mode! Vertices: ${modelData.vertices.length/4}, 
                Faces: ${modelData.indices.length/3}</p>`;
                
        } catch (error) {
            console.error('Failed to load model:', error);
            document.getElementById('info').innerHTML = 
                '<p style="color: red;">Failed to load model: ' + error.message + '</p>';
        }
    }

    createBuffers() {
        if (!this.modelData) return;
        
        console.log('Creating buffers...');
        
        // Convert 4D vertices to 3D vertices (remove w component)
        const vertices3D = new Float32Array(this.modelData.vertices.length / 4 * 3);
        const normals3D = new Float32Array(this.modelData.normals.length / 4 * 3);
        
        for (let i = 0; i < this.modelData.vertices.length / 4; i++) {
            vertices3D[i * 3] = this.modelData.vertices[i * 4];     // x
            vertices3D[i * 3 + 1] = this.modelData.vertices[i * 4 + 1]; // y
            vertices3D[i * 3 + 2] = this.modelData.vertices[i * 4 + 2]; // z
        }
        
        for (let i = 0; i < this.modelData.normals.length / 4; i++) {
            normals3D[i * 3] = this.modelData.normals[i * 4];     // x
            normals3D[i * 3 + 1] = this.modelData.normals[i * 4 + 1]; // y
            normals3D[i * 3 + 2] = this.modelData.normals[i * 4 + 2]; // z
        }
        
        console.log('Converted to 3D vertices:', vertices3D.length / 3);
        console.log('Converted to 3D normals:', normals3D.length / 3);

        // Create vertex buffer (3D vertices) - ensure size is multiple of 4
        const vertexBufferSize = Math.ceil(vertices3D.byteLength / 4) * 4;
        console.log('Vertex buffer size:', vertices3D.byteLength, 'aligned to:', vertexBufferSize);
        this.vertexBuffer = this.device.createBuffer({
            size: vertexBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        console.log('Writing vertex buffer...');
        this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices3D);

        // Create normal buffer (3D normals) - ensure size is multiple of 4
        const normalBufferSize = Math.ceil(normals3D.byteLength / 4) * 4;
        console.log('Normal buffer size:', normals3D.byteLength, 'aligned to:', normalBufferSize);
        this.normalBuffer = this.device.createBuffer({
            size: normalBufferSize,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        console.log('Writing normal buffer...');
        this.device.queue.writeBuffer(this.normalBuffer, 0, normals3D);

        // Create barycentric coordinate buffer for wireframe overlay
        if (this.modelData.barycoords) {
            // Convert 3D barycentric coordinates
            const barycoords3D = new Float32Array(this.modelData.barycoords.length);
            for (let i = 0; i < this.modelData.barycoords.length; i++) {
                barycoords3D[i] = this.modelData.barycoords[i];
            }
            
            // Ensure buffer size is multiple of 4 bytes
            const barycoordBufferSize = Math.ceil(barycoords3D.byteLength / 4) * 4;
            console.log('Barycoord buffer size:', barycoords3D.byteLength, 'aligned to:', barycoordBufferSize);
            
            this.barycoordBuffer = this.device.createBuffer({
                size: barycoordBufferSize,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
            console.log('Writing barycoord buffer...');
            this.device.queue.writeBuffer(this.barycoordBuffer, 0, barycoords3D);
        }

        // Create index buffer (convert to uint16 for compatibility)
        const indices16 = new Uint16Array(this.modelData.indices.length);
        for (let i = 0; i < this.modelData.indices.length; i++) {
            indices16[i] = this.modelData.indices[i];
        }
        
        // Ensure buffer size is multiple of 4 bytes
        const indexBufferSize = Math.ceil(indices16.byteLength / 4) * 4;
        console.log('Index buffer size:', indices16.byteLength, 'aligned to:', indexBufferSize);
        
        // Create padded index data to match buffer size
        const paddedIndices = new Uint16Array(indexBufferSize / 2); // 2 bytes per uint16
        paddedIndices.set(indices16, 0);
        
        this.indexBuffer = this.device.createBuffer({
            size: indexBufferSize,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        console.log('Writing index buffer...');
        this.device.queue.writeBuffer(this.indexBuffer, 0, paddedIndices);

        // Create uniform buffer (MVP matrix + camera position + lighting parameters)
        this.uniformBuffer = this.device.createBuffer({
            size: 96, // 24 floats = 96 bytes (16 + 3 + 5 floats with proper alignment)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        console.log('All buffers created successfully');
    }

    createRenderPipeline() {
        console.log('Creating render pipeline with Phong lighting');

        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({
                    code: this.getVertexShader()
                }),
                        buffers: [
                            {
                                arrayStride: 12, // 3 floats * 4 bytes (x, y, z)
                                attributes: [
                                    { shaderLocation: 0, offset: 0, format: 'float32x3' } // position
                                ]
                            },
                            {
                                arrayStride: 12, // 3 floats * 4 bytes (x, y, z)
                                attributes: [
                                    { shaderLocation: 1, offset: 0, format: 'float32x3' } // normal
                                ]
                            }
                        ]
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: this.getFragmentShader()
                }),
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat()
                }]
            },
            primitive: {
                topology: 'triangle-list', // Always use triangles, wireframe is handled in shader
                cullMode: 'back'
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus'
            }
        });
        console.log('Render pipeline created successfully');
    }

    createWireframePipeline() {
        console.log('Creating wireframe overlay pipeline');
        
        this.wireframePipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({
                    code: this.getWireframeVertexShader()
                }),
                buffers: [
                    {
                        arrayStride: 12, // 3 floats * 4 bytes (x, y, z)
                        attributes: [
                            { shaderLocation: 0, offset: 0, format: 'float32x3' } // position
                        ]
                    },
                    {
                        arrayStride: 12, // 3 floats * 4 bytes (u, v, w)
                        attributes: [
                            { shaderLocation: 1, offset: 0, format: 'float32x3' } // barycentric coordinates
                        ]
                    }
                ]
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: this.getWireframeFragmentShader()
                }),
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat()
                }]
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back'
            },
            depthStencil: {
                depthWriteEnabled: true, // Write depth for proper alignment
                depthCompare: 'less', // Standard depth testing
                format: 'depth24plus'
            }
        });
        console.log('Wireframe pipeline created successfully');
    }

    getVertexShader() {
        return `
            struct Uniforms {
                mvpMatrix: mat4x4<f32>,
                cameraPos: vec3<f32>,
                kd: f32,
                ks: f32,
                shininess: f32,
                le: f32,
                la: f32,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) worldPos: vec3<f32>,
                @location(1) normal: vec3<f32>,
            }

            @vertex
            fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> VertexOutput {
                var output: VertexOutput;
                output.position = uniforms.mvpMatrix * vec4<f32>(position, 1.0);
                output.worldPos = position;
                output.normal = normalize(normal);
                return output;
            }
        `;
    }

    getWireframeVertexShader() {
        return `
            struct Uniforms {
                mvpMatrix: mat4x4<f32>,
                cameraPos: vec3<f32>,
                kd: f32,
                ks: f32,
                shininess: f32,
                le: f32,
                la: f32,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) barycentric: vec3<f32>,
            }

            @vertex
            fn vs_main(@location(0) position: vec3<f32>, @location(1) barycentric: vec3<f32>) -> VertexOutput {
                var output: VertexOutput;
                // Slight offset to ensure wireframe renders on top
                var pos = uniforms.mvpMatrix * vec4<f32>(position, 1.0);
                pos.z -= 0.0001; // Small depth offset
                output.position = pos;
                output.barycentric = barycentric;
                return output;
            }
        `;
    }

    getFragmentShader() {
        return `
            struct Uniforms {
                mvpMatrix: mat4x4<f32>,
                cameraPos: vec3<f32>,
                kd: f32,
                ks: f32,
                shininess: f32,
                le: f32,
                la: f32,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) worldPos: vec3<f32>,
                @location(1) normal: vec3<f32>,
            }

            @fragment
            fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
                // Re-normalize interpolated vectors (important!)
                let normal = normalize(input.normal);
                let worldPos = input.worldPos;
                
                // Directional light with direction le = (0, 0, -1)
                // Light direction vector towards light: l = -le = (0, 0, 1)
                let lightDir = vec3<f32>(0.0, 0.0, 1.0);
                let lightEmission = vec3<f32>(uniforms.le, uniforms.le, uniforms.le); // White light
                let ambientLight = vec3<f32>(uniforms.la, uniforms.la, uniforms.la); // White ambient
                
                // View direction (towards camera) - re-normalize
                let viewDir = normalize(uniforms.cameraPos - worldPos);
                
                // Material colors
                let diffuseColor = vec3<f32>(0.8, 0.3, 0.3); // Red diffuse
                let specularColor = vec3<f32>(1.0, 1.0, 1.0); // White specular
                
                // 1. Ambient component: ka * La (where ka = kd)
                let ambient = uniforms.kd * diffuseColor * ambientLight;
                
                // 2. Diffuse component: kd * Li * max(n · l, 0)
                let nDotL = max(dot(normal, lightDir), 0.0);
                let diffuse = uniforms.kd * diffuseColor * lightEmission * nDotL;
                
                // 3. Specular component: ks * Li * max(r · v, 0)^s
                let reflectDir = reflect(-lightDir, normal);
                let rDotV = max(dot(reflectDir, viewDir), 0.0);
                let specular = uniforms.ks * specularColor * lightEmission * pow(rDotV, uniforms.shininess);
                
                // Final color = ambient + diffuse + specular
                let finalColor = ambient + diffuse + specular;
                
                return vec4<f32>(finalColor, 1.0);
            }
        `;
    }

    getWireframeFragmentShader() {
        return `
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) barycentric: vec3<f32>,
            }

            @fragment
            fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
                // Wireframe using barycentric coordinates
                let edgeDistance = min(min(input.barycentric.x, input.barycentric.y), input.barycentric.z);
                let lineWidth = 0.025; // Increased line width for better visibility
                let alpha = 1.0 - smoothstep(0.0, lineWidth, edgeDistance);
                
                if (alpha < 0.1) {
                    discard;
                }
                
                return vec4<f32>(0.0, 0.0, 0.0, alpha); // Black wireframe overlay
            }
        `;
    }

    createMatrices() {
        // Use MV.js functions
        const projection = perspective(45, this.canvas.width / this.canvas.height, 0.1, 100);
        
        // Orbiting camera (only if rotation is enabled)
        const time = this.isRotating ? (Date.now() - this.rotationStartTime) * 0.001 : 0;
        const angle = time * 0.5 + this.rotationOffset;
        const radius = 6.0; // Move camera much further back
        const cameraX = radius * Math.cos(angle);
        const cameraY = 0;
        const cameraZ = radius * Math.sin(angle);
        const view = lookAt(vec3(cameraX, cameraY, cameraZ), vec3(0, 0, 0), vec3(0, 1, 0));
        
        const model = scalem(2.0, 2.0, 2.0); // Scale up the model
        
        const mvp = mult(projection, mult(view, model));
        
        console.log('Camera position:', cameraX, cameraY, cameraZ);
        console.log('MVP matrix:', flatten(mvp));

        return { mvpMatrix: flatten(mvp), cameraPos: [cameraX, cameraY, cameraZ] };
    }

    render() {
        if (!this.modelData || !this.renderPipeline) {
            console.log('Cannot render: modelData =', !!this.modelData, 'renderPipeline =', !!this.renderPipeline);
            return;
        }

        console.log('Rendering model...');
        const { mvpMatrix, cameraPos } = this.createMatrices();

        // Update uniform buffer (week4 approach)
        const uniformData = new Float32Array(24); // 24 floats = 96 bytes
        
        // MVP matrix at offset 0-15
        uniformData.set(mvpMatrix, 0);
        
        // Camera position at offset 16-18
        uniformData.set(cameraPos, 16);
        
        // Lighting parameters at offset 19-23
        uniformData[19] = this.lightingParams.kd;
        uniformData[20] = this.lightingParams.ks;
        uniformData[21] = this.lightingParams.shininess;
        uniformData[22] = this.lightingParams.le;
        uniformData[23] = this.lightingParams.la;

        console.log('Uniform data size:', uniformData.length);
        console.log('Camera position:', cameraPos);

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
        console.log('Uniform buffer updated');

        // Create depth texture
        const depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
        console.log('Depth texture created:', this.canvas.width, 'x', this.canvas.height);

        // Create command encoder
        const commandEncoder = this.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.9, g: 0.9, b: 1.0, a: 1.0 }, // Light blue background
                loadOp: 'clear',
                storeOp: 'store'
            }],
            depthStencilAttachment: {
                view: depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store'
            }
        });
        console.log('Render pass started');

        // Draw solid model first
        console.log('Drawing solid model...');
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setVertexBuffer(0, this.vertexBuffer);
        renderPass.setVertexBuffer(1, this.normalBuffer);
        renderPass.setIndexBuffer(this.indexBuffer, 'uint16');
        
        const bindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: { buffer: this.uniformBuffer }
            }]
        });
        renderPass.setBindGroup(0, bindGroup);
        renderPass.drawIndexed(this.modelData.indices.length);

        // Draw wireframe overlay if enabled
        if (this.isWireframe && this.barycoordBuffer && this.wireframePipeline) {
            console.log('Drawing wireframe overlay...');
            renderPass.setPipeline(this.wireframePipeline);
            renderPass.setVertexBuffer(0, this.vertexBuffer);
            renderPass.setVertexBuffer(1, this.barycoordBuffer);
            renderPass.setIndexBuffer(this.indexBuffer, 'uint16');
            
            const wireframeBindGroup = this.device.createBindGroup({
                layout: this.wireframePipeline.getBindGroupLayout(0),
                entries: [{
                    binding: 0,
                    resource: { buffer: this.uniformBuffer }
                }]
            });
            renderPass.setBindGroup(0, wireframeBindGroup);
            renderPass.drawIndexed(this.modelData.indices.length);
        }
        renderPass.end();

        // Submit command buffer
        this.device.queue.submit([commandEncoder.finish()]);
        console.log('Render submitted');
        
        // Continue animation for orbiting camera
        requestAnimationFrame(() => this.render());
    }
}

// Initialize the renderer when the page loads
window.addEventListener('load', () => {
    new WebGPURenderer();
});
