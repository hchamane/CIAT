
  (function ($) {
  
  "use strict";

    // MENU
    $('.navbar-collapse a').on('click',function(){
      $(".navbar-collapse").collapse('hide');
    });
    
    // CUSTOM LINK
    $('.smoothscroll').click(function(){
      var el = $(this).attr('href');
      var elWrapped = $(el);
      var header_height = $('.navbar').height();
  
      scrollToDiv(elWrapped,header_height);
      return false;
  
      function scrollToDiv(element,navheight){
        var offset = element.offset();
        var offsetTop = offset.top;
        var totalScroll = offsetTop-navheight;
  
        $('body,html').animate({
        scrollTop: totalScroll
        }, 300);
      }
    });

    $(window).on('scroll', function(){
      function isScrollIntoView(elem, index) {
        var docViewTop = $(window).scrollTop();
        var docViewBottom = docViewTop + $(window).height();
        var elemTop = $(elem).offset().top;
        var elemBottom = elemTop + $(window).height()*.5;
        if(elemBottom <= docViewBottom && elemTop >= docViewTop) {
          $(elem).addClass('active');
        }
        if(!(elemBottom <= docViewBottom)) {
          $(elem).removeClass('active');
        }
        var MainTimelineContainer = $('#vertical-scrollable-timeline')[0];
        var MainTimelineContainerBottom = MainTimelineContainer.getBoundingClientRect().bottom - $(window).height()*.5;
        $(MainTimelineContainer).find('.inner').css('height',MainTimelineContainerBottom+'px');
      }
      var timeline = $('#vertical-scrollable-timeline li');
      Array.from(timeline).forEach(isScrollIntoView);
    });
  
  })(window.jQuery);

document.addEventListener('DOMContentLoaded', function() {
    console.log('Train model page loaded');
    
    // Force display all cards and content
    document.querySelectorAll('.card').forEach(function(card) {
        card.style.display = 'block';
        card.style.visibility = 'visible';
        card.style.overflow = 'visible';
        console.log('Made card visible:', card);
    });
    
    // Log all form elements to check if they're properly loaded
    document.querySelectorAll('form input, form select, form button').forEach(function(element) {
        console.log('Form element:', element.name || element.id, 'Type:', element.type);
    });
    
    // Add margin and spacing to ensure elements don't overlap
    document.querySelectorAll('.row').forEach(function(row) {
        row.style.marginBottom = '30px';
        console.log('Added margin to row');
    });
    
    // Ensure all card-bodies have minimum height
    document.querySelectorAll('.card-body').forEach(function(cardBody) {
        cardBody.style.minHeight = '100px';
        console.log('Set minimum height for card body');
    });
    
    // Validate that all required scripts are loaded
    ['jQuery', 'Bootstrap'].forEach(function(lib) {
        if (lib === 'jQuery' && typeof jQuery !== 'undefined') {
            console.log('✅ jQuery is loaded');
        } else if (lib === 'Bootstrap' && typeof bootstrap !== 'undefined') {
            console.log('✅ Bootstrap is loaded');
        } else {
            console.warn('❌ ' + lib + ' is NOT loaded');
        }
    });
    
    // Check if model_trained variable is properly set in the template
    const trainedStatusElements = document.querySelectorAll('.model-status.trained, .status-box.trained');
    const untrainedStatusElements = document.querySelectorAll('.model-status.untrained, .status-box.untrained');
    
    console.log('Trained status elements:', trainedStatusElements.length);
    console.log('Untrained status elements:', untrainedStatusElements.length);
    
    // Add event listener to train button
    const trainButton = document.getElementById('trainButton');
    if (trainButton) {
        console.log('Train button found, adding event listener');
        trainButton.addEventListener('click', function() {
            console.log('Train button clicked');
        });
    } else {
        console.warn('Train button not found');
    }
    
    // Add CSS properties to ensure basic styling even if stylesheets fail
    const basicStyles = {
        '.card': {
            display: 'block',
            backgroundColor: 'white',
            border: '1px solid #ddd',
            borderRadius: '8px',
            marginBottom: '20px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        },
        '.card-header': {
            padding: '15px',
            borderBottom: '1px solid #ddd',
            backgroundColor: '#f8f9fa'
        },
        '.card-body': {
            padding: '15px'
        },
        '.btn': {
            display: 'inline-block',
            padding: '8px 16px',
            backgroundColor: '#13547a',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            textDecoration: 'none',
            cursor: 'pointer'
        },
        '.step-number': {
            display: 'inline-block',
            width: '30px',
            height: '30px',
            backgroundColor: '#13547a',
            color: 'white',
            borderRadius: '50%',
            textAlign: 'center',
            lineHeight: '30px',
            marginRight: '10px'
        }
    };
    
    // Apply emergency styles if needed
    for (const selector in basicStyles) {
        const elements = document.querySelectorAll(selector);
        const styles = basicStyles[selector];
        
        elements.forEach(element => {
            for (const property in styles) {
                element.style[property] = styles[property];
            }
        });
    }
    
    console.log('Emergency styles applied');
});

// Add this script to enhance form visibility and interaction
// You can include this in a custom.js file or add it to your assess.html template

document.addEventListener('DOMContentLoaded', function() {
    // Enhance range sliders
    enhanceRangeSliders();
    
    // Improve dropdown visibility
    enhanceDropdowns();
    
    // Make checkboxes more visible
    enhanceCheckboxes();
    
    // Add better hover effects to form elements
    addHoverEffects();
    
    // Form validation enhancements
    enhanceFormValidation();
});

// Function to enhance range sliders with better visuals
function enhanceRangeSliders() {
    const rangeSliders = document.querySelectorAll('input[type="range"]');
    
    rangeSliders.forEach(slider => {
        // Add visual indicator class
        slider.classList.add('enhanced-slider');
        
        // Ensure the output element shows the initial value
        const output = slider.nextElementSibling;
        if (output && output.tagName.toLowerCase() === 'output') {
            // Set initial value
            if (slider.id === 'virtual_team_ratio') {
                output.value = slider.value + '%';
            } else {
                output.value = slider.value;
            }
            
            // Make sure output is visible
            output.style.display = 'inline-block';
        }
        
        // Add color indicator based on value
        updateSliderColor(slider);
        
        // Update color when value changes
        slider.addEventListener('input', function() {
            updateSliderColor(this);
        });
    });
}

// Function to color the slider based on its value
function updateSliderColor(slider) {
    // Calculate percentage of slider value
    const min = parseInt(slider.min) || 0;
    const max = parseInt(slider.max) || 100;
    const value = parseInt(slider.value) || 0;
    const percentage = ((value - min) / (max - min)) * 100;
    
    // Set background gradient to show progress
    slider.style.background = `linear-gradient(to right, #13547a 0%, #13547a ${percentage}%, #d7e9f2 ${percentage}%, #d7e9f2 100%)`;
}

// Function to enhance dropdown menus
function enhanceDropdowns() {
    const dropdowns = document.querySelectorAll('select, .form-select');
    
    dropdowns.forEach(dropdown => {
        // Add visual indicator class
        dropdown.classList.add('enhanced-dropdown');
        
        // Add icon indicator if not already present in styling
        if (!dropdown.style.backgroundImage) {
            dropdown.style.backgroundImage = "url('data:image/svg+xml;utf8,<svg fill=\"%2313547a\" height=\"24\" viewBox=\"0 0 24 24\" width=\"24\" xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M7 10l5 5 5-5z\"/><path d=\"M0 0h24v24H0z\" fill=\"none\"/></svg>')";
            dropdown.style.backgroundRepeat = 'no-repeat';
            dropdown.style.backgroundPosition = 'right 10px center';
        }
        
        // Add animation for dropdown
        dropdown.addEventListener('focus', function() {
            this.style.boxShadow = '0 0 0 0.25rem rgba(19, 84, 122, 0.25)';
        });
        
        dropdown.addEventListener('blur', function() {
            this.style.boxShadow = 'none';
        });
    });
}

// Function to enhance checkboxes
function enhanceCheckboxes() {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    
    checkboxes.forEach(checkbox => {
        // Add visual indicator class
        checkbox.classList.add('enhanced-checkbox');
        
        // Enhance the checkbox container
        const label = checkbox.closest('label') || checkbox.nextElementSibling;
        if (label) {
            label.style.cursor = 'pointer';
            const container = checkbox.closest('.form-check');
            if (container) {
                container.style.transition = 'background-color 0.2s';
                
                container.addEventListener('mouseenter', function() {
                    this.style.backgroundColor = 'rgba(128, 208, 199, 0.1)';
                });
                
                container.addEventListener('mouseleave', function() {
                    this.style.backgroundColor = 'transparent';
                });
            }
        }
    });
}

// Function to add hover effects to form elements
function addHoverEffects() {
    // Add hover effect to form sections
    const formSections = document.querySelectorAll('.row');
    formSections.forEach(section => {
        if (section.querySelector('h4')) {
            section.style.transition = 'transform 0.2s';
            section.addEventListener('mouseenter', function() {
                this.style.transform = 'translateX(5px)';
            });
            section.addEventListener('mouseleave', function() {
                this.style.transform = 'translateX(0)';
            });
        }
    });
    
    // Add hover effect to the submit button
    const submitButton = document.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
        });
        
        submitButton.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 3px 6px rgba(0,0,0,0.15)';
        });
    }
}

// Function to enhance form validation
function enhanceFormValidation() {
    const form = document.querySelector('form');
    if (!form) return;
    
    // Add validation styling
    form.addEventListener('submit', function(event) {
        if (!form.checkValidity()) {
            event.preventDefault();
            event.stopPropagation();
            
            // Highlight invalid fields with animation
            const invalidFields = form.querySelectorAll(':invalid');
            invalidFields.forEach(field => {
                field.style.animation = 'shake 0.5s';
                setTimeout(() => {
                    field.style.animation = '';
                }, 500);
            });
            
            // Scroll to first invalid field
            if (invalidFields.length > 0) {
                invalidFields[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
        
        form.classList.add('was-validated');
    });
    
    // Add keyframe animation for shake effect
    if (!document.getElementById('validation-animations')) {
        const style = document.createElement('style');
        style.id = 'validation-animations';
        style.textContent = `
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
                20%, 40%, 60%, 80% { transform: translateX(5px); }
            }
        `;
        document.head.appendChild(style);
    }
    
    // Validate countries selection
    const countryCheckboxes = document.querySelectorAll('input[name="countries"]');
    if (countryCheckboxes.length > 0) {
        const countriesContainer = document.getElementById('countries-container');
        if (countriesContainer) {
            // Add visual indication that countries are required
            const label = countriesContainer.querySelector('.form-label');
            if (label) {
                if (!label.textContent.includes('*')) {
                    label.textContent += ' *';
                }
                label.style.color = '#13547a';
                label.style.fontWeight = 'bold';
            }
            
            // Add a reminder message
            const helpText = document.createElement('small');
            helpText.className = 'form-text text-muted';
            helpText.textContent = 'Please select at least one country involved in the project';
            countriesContainer.appendChild(helpText);
        }
    }
}
