// frontend/src/components/PriceChart.js

import React from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ResponsiveContainer } from 'recharts';

const PriceChart = ({ data, chartType = 'line', dataKey = 'Price', xKey = 'Date', title = '' }) => {
    return (
        <div>
            {title && <h4 style={{ textAlign: 'center', marginBottom: '15px' }}>{title}</h4>}
            <ResponsiveContainer width="100%" height={300}>
                {chartType === 'line' ? (
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey={xKey} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey={dataKey} stroke="#8884d8" />
                    </LineChart>
                ) : (
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey={xKey} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey={dataKey} fill="#8884d8" />
                    </BarChart>
                )}
            </ResponsiveContainer>
        </div>
    );
}

export default PriceChart;
