# Usage Tips for Best 3D Reconstruction Results

## Image Requirements

### File Size and Format
- **Total upload limit**: 64MB for web interface
- **Supported formats**: JPG, PNG, BMP
- **Recommended**: Use JPG format for smaller file sizes
- **Image resolution**: 800-2000 pixels on longest side for best results

### Image Quality Guidelines

#### ✅ Good Images:
- **Sharp and well-focused** - avoid motion blur
- **Good lighting** - avoid very dark or overexposed areas  
- **Rich texture** - objects with patterns, text, or surface details
- **60-80% overlap** between consecutive views
- **Multiple viewpoints** - move around the object/scene
- **Stable lighting** - consistent illumination across images

#### ❌ Avoid:
- **Blurry or shaky images**
- **Very dark or bright images**
- **Smooth/reflective surfaces** (glass, mirrors, polished metal)
- **Uniform textures** (blank walls, solid colors)
- **Moving objects** in the scene
- **Dramatic lighting changes** between shots

## Shooting Techniques

### For Objects:
1. **Place object on textured surface** (newspaper, patterned cloth)
2. **Circle around the object** taking photos every 15-30 degrees
3. **Vary height slightly** - some higher, some lower angles
4. **Keep consistent distance** from the object
5. **Use tripod or steady hands** to avoid blur

### For Scenes/Buildings:
1. **Walk in arc or circle** around the subject
2. **Take overlapping photos** with 60-80% overlap
3. **Include foreground and background** for depth cues
4. **Avoid people walking** through the scene
5. **Take photos at consistent time** to maintain lighting

## Web Interface Usage

### Upload Process:
1. **Select 2-10 images** (2 minimum for stereo reconstruction)
2. **Check total file size** stays under 64MB
3. **Review image preview** before processing
4. **Click "Start 3D Reconstruction"**

### Understanding Results:
- **Feature Matches**: Shows detected keypoints and matches between images
- **3D Points**: Number of reconstructed 3D points (more is usually better)
- **Mean Depth**: Average distance of points from camera
- **Reprojection Error**: Lower values indicate better reconstruction quality

## Troubleshooting Common Issues

### "Not enough matches found"
- **Solution**: Use images with more texture and detail
- **Try**: Different angles, closer shots, better lighting

### "Poor reconstruction quality"
- **Solution**: Ensure better image overlap and quality
- **Try**: More images, consistent lighting, sharper focus

### "Files too large"
- **Solution**: Resize images or use fewer images
- **Try**: Reduce resolution to 1200px max, use JPG format

### "Reconstruction failed"
- **Solution**: Check image quality and overlap
- **Try**: Different image pairs, add more texture to scene

## Example Image Sets

### Good Example - Textured Object:
```
├── object_01.jpg  (front view)
├── object_02.jpg  (30° right)
├── object_03.jpg  (60° right) 
├── object_04.jpg  (90° right)
├── object_05.jpg  (120° right)
```

### Good Example - Building/Scene:
```
├── building_01.jpg  (left side)
├── building_02.jpg  (left-front angle)
├── building_03.jpg  (front view)
├── building_04.jpg  (right-front angle)
├── building_05.jpg  (right side)
```

## Advanced Tips

### Camera Settings (if using DSLR):
- **Fixed focal length** (avoid zoom changes)
- **Manual focus** to avoid focus hunting
- **Aperture f/8-f/11** for good depth of field
- **ISO 100-400** for minimal noise
- **Fast shutter** to avoid motion blur

### Lighting:
- **Soft, even lighting** works best
- **Avoid harsh shadows** or direct sunlight
- **Overcast days** provide excellent natural lighting
- **Indoor**: Use multiple light sources to reduce shadows

### Post-Processing:
- **Color correction** - ensure consistent colors across images
- **Resize large images** to 1600px max for faster processing
- **Remove duplicates** - avoid nearly identical viewpoints

Remember: Good input images are the key to successful 3D reconstruction!