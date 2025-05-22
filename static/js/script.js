// Animate form inputs on focus
document.querySelectorAll('input, select').forEach(element => {
    element.addEventListener('focus', function() {
        this.style.transform = 'scale(1.02)';
    });
    element.addEventListener('blur', function() {
        this.style.transform = 'scale(1)';
    });
});

// Animate the result section when it appears
document.addEventListener('DOMContentLoaded', function() {
    const result = document.getElementById('result');
    if (result) {
        result.style.opacity = '0';
        setTimeout(() => {
            result.style.opacity = '1';
        }, 100);
    }
});