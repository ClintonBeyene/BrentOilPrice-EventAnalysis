// src/components/Header.js
import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';

const Header = () => {
    return (
        <header className="header">
            <h1>Brent Oil Price Dashboard</h1>
            <nav>
                <ul className="nav">
                    <li className="nav-item"><Link className="nav-link" to="/">Home</Link></li>
                    <li className="nav-item"><Link className="nav-link" to="/dashboard">Analysis</Link></li>
                    <li className="nav-item"><Link className="nav-link" to="/about">About</Link></li> {/* Link to About */}
                </ul>
            </nav>
        </header>
    );
}

export default Header;