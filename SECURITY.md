# Security Policy

## Data Privacy and Security

### 🔒 Local Processing Only
- **No Data Transmission**: All image processing and reconstruction happens locally on your machine
- **No External Services**: No data is sent to external servers or cloud services
- **No User Tracking**: We do not collect, store, or transmit any user data or analytics

### 🛡️ File Handling Security
- **Temporary Files**: Upload files are automatically cleaned up after processing
- **Path Validation**: All file operations include path validation to prevent directory traversal
- **Size Limits**: File upload size is limited to prevent resource exhaustion (512MB total)
- **Format Validation**: Only accepted image formats (JPG, PNG, BMP) are processed

### 🔐 Web Interface Security
- **Input Sanitization**: All user inputs are validated and sanitized
- **Memory Management**: Automatic cleanup of temporary directories and files
- **Error Handling**: Sensitive system information is not exposed in error messages
- **Session Management**: No user sessions or persistent data storage

### 📁 File System Security
- **Restricted Access**: File operations are restricted to designated temporary directories
- **Automatic Cleanup**: Downloaded files are automatically removed after 1 hour
- **No Persistence**: No user data is permanently stored on the server

## Reporting Security Issues

If you discover a security vulnerability, please report it by:

1. **Email**: Send details to [security@yourproject.com] (replace with actual contact)
2. **GitHub Issues**: Create a private security advisory
3. **Include**: Steps to reproduce, potential impact, and suggested fixes

### Please Do Not:
- Disclose the vulnerability publicly until it has been addressed
- Access or modify data that doesn't belong to you
- Attempt to perform denial-of-service attacks

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | ✅ Fully supported |
| < 1.0   | ❌ Not supported   |

## Security Best Practices for Users

### When Using This Software:
1. **Keep Software Updated**: Always use the latest version
2. **Trusted Sources**: Only process images from trusted sources
3. **Local Network**: Run on trusted networks only
4. **Resource Monitoring**: Monitor system resources during large reconstructions
5. **File Cleanup**: Verify temporary files are cleaned up after use

### Production Deployment (Not Recommended):
This software is designed for local development and research use. If you must deploy it:
- Use a reverse proxy (nginx, Apache)
- Implement proper authentication
- Use HTTPS/TLS encryption
- Monitor resource usage
- Implement rate limiting
- Regular security audits

## Known Limitations

1. **Resource Usage**: Large image sets may consume significant memory
2. **Temporary Files**: Temporary files may persist if process is forcefully terminated
3. **Error Messages**: Some error messages may reveal internal paths (development mode)

## Contact

For security-related questions or concerns:
- Create an issue on GitHub
- Follow responsible disclosure practices
- Allow reasonable time for fixes before public disclosure

---

**Last Updated**: 2025-06-18