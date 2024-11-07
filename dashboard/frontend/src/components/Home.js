// src/components/Home.js
import React from 'react';
import './Home.css';

const Home = () => {
    return (
        <div className="home">
            <h1>Welcome to the Brent Oil Price Dashboard</h1>
            <p>
                This dashboard provides comprehensive insights into Brent oil prices, including historical trends, event impacts, and statistical analyses.
            </p>
            <h2>Features</h2>
            <ul>
                <li>View historical Brent oil price trends</li>
                <li>Analyze the impact of significant events on oil prices</li>
                <li>Export data for further analysis</li>
                <li>Access detailed statistics and visualizations</li>
            </ul>
            <h2>Get Started</h2>
            <p>
                Use the navigation menu to explore the various features of the dashboard. Click on the links to dive into the data and insights.
            </p>
        </div>
    );
}

export default Home;