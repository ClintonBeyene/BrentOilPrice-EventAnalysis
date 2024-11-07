// src/components/About.js
import React from 'react';
import './About.css'; // Import the CSS file for styling

const About = () => {
    return (
        <div className="about">
            <h2>About Me</h2>
            <p>
                I am a passionate Data Scientist with experience in Data Analytics and Model Building.
                Feel free to reach out to me through the links below.
            </p>
            <div className="contact">
                <span className="contact-item">Contact Me</span>
                <div className="links">
                    <a href="https://github.com/ClintonBeyene" target="_blank" rel="noopener noreferrer" className="link">GitHub</a>
                    <a href="https://www.linkedin.com/in/clinton-beyene" target="_blank" rel="noopener noreferrer" className="link">LinkedIn</a>
                </div>
            </div>
        </div>
    );
};

export default About;