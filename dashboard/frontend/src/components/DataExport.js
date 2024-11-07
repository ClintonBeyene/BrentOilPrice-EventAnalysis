// frontend/src/components/DataExport.js

import React from 'react';
import { saveAs } from 'file-saver';

const DataExport = ({ data }) => {
    const exportToCSV = () => {
        const csvContent = "data:text/csv;charset=utf-8," + 
            data.map(e => `${e.Date},${e.Price}`).join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "brent_oil_prices.csv"); document.body.appendChild(link); link.click(); document.body.removeChild(link); };
        
        return (
            <button onClick={exportToCSV}>Export Data as CSV</button>
        );
}

export default DataExport;     
