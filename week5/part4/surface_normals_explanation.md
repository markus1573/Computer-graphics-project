# Surface Normals in Triangle Mesh Rendering

## What are Surface Normals?

Surface normals are vectors perpendicular to surfaces that define their orientation. They are crucial for lighting calculations and determining how surfaces appear when rendered.

## How Surface Normals are Obtained

### 1. From OBJ Files
- OBJ files can contain explicit normal definitions using `vn` commands
- These normals are typically pre-calculated and provide smooth shading

### 2. Calculated from Triangle Geometry
When normals aren't provided, they are computed from triangle vertices:
1. **Create edge vectors** from triangle vertices
2. **Calculate cross product** of the edge vectors
3. **Normalize** the result to unit length

The cross product gives a vector perpendicular to the triangle surface.

## How Surface Normals are Used

### In Lighting Calculations
Normals are essential for:
- **Diffuse lighting**: `dot(normal, lightDirection)` determines light intensity
- **Specular lighting**: `reflect(-lightDirection, normal)` calculates reflection direction
- **View-dependent effects**: Normals help determine viewing angles for highlights

### In the Rendering Pipeline
1. **Vertex shader**: Passes normals from vertices
2. **Rasterization**: Interpolates normals across triangle surfaces
3. **Fragment shader**: Uses interpolated normals for lighting calculations
4. **Re-normalization**: Interpolated normals must be normalized for accuracy

## Relationship to Surface Smoothness

### Flat Shading (Faceted Appearance)
- Uses **face normals** - one normal per triangle
- All vertices of a triangle share the same normal
- Results in **sharp, angular edges** between triangles
- Good for hard surfaces like buildings or machinery

### Smooth Shading (Smooth Appearance)
- Uses **vertex normals** - averaged across adjacent faces
- Normals are interpolated across triangle surfaces
- Results in **smooth, curved transitions**
- Good for organic shapes like characters or natural objects

### Visual Comparison

| Flat Shading | Smooth Shading |
|--------------|----------------|
| Face normals | Vertex normals |
| Faceted appearance | Smooth appearance |
| Sharp edges | Soft transitions |
| Lower cost | Higher cost |

## Key Concepts

- **Normal interpolation**: GPU interpolates normals from vertices to fragments
- **Re-normalization**: Essential for accurate lighting after interpolation
- **Face vs vertex normals**: Determines flat vs smooth appearance
- **Lighting dependency**: Normals directly affect how light interacts with surfaces

## Summary

Surface normals control both the **visual appearance** (flat vs smooth) and **lighting quality** of rendered surfaces. The choice between face normals (flat shading) and vertex normals (smooth shading) depends on the desired visual effect and the nature of the 3D model.
