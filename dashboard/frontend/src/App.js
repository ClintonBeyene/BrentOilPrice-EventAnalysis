// src/App.js
import React from 'react';
import './App.css';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import Home from './components/Home';
import About from './components/About'; // Ensure this is a default export
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

function App() {
    return (
        <Router>
            <div>
                <Header />
                <main>
                    <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/dashboard" element={<Dashboard />} />
                        <Route path="/about" element={<About />} /> {/* Add About route */}
                    </Routes>
                </main>
            </div>
        </Router>
    );
}

export default App;