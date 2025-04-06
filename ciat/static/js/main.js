// Cultural Impact Assessment Tool (CIAT) - Main JavaScript

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Flash message auto-close
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(function(message) {
        // Add close button
        const closeButton = document.createElement('button');
        closeButton.className = 'close';
        closeButton.innerHTML = '&times;';
        closeButton.style.float = 'right';
        closeButton.style.cursor = 'pointer';
        closeButton.style.border = 'none';
        closeButton.style.background = 'none';
        closeButton.style.fontSize = '20px';
        closeButton.onclick = function() {
            message.style.display = 'none';
        };
        message.prepend(closeButton);
        
        // Auto-close after 5 seconds
        setTimeout(function() {
            message.style.display = 'none';
        }, 5000);
    });
    
    // Rest of JS in \ciat\static\js 
});
