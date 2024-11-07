import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
    LineElement,
    PointElement,
    Filler,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { Table } from 'reactstrap';
import annotationPlugin from 'chartjs-plugin-annotation';
import { Line } from 'react-chartjs-2'; // Import Line chart

// Register the necessary components with ChartJS
ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    LineElement,
    PointElement,
    Title,
    Tooltip,
    Legend,
    Filler,
    annotationPlugin // Register the annotation plugin
);

const EventImpact = () => {
    const [eventData, setEventData] = useState([]);
    const [priceTrends, setPriceTrends] = useState({ labels: [], prices: [] });
    const [eventDates, setEventDates] = useState([]); // State to store event dates

    useEffect(() => {
        const fetchEventData = async () => {
            try {
                const response = await axios.get('http://localhost:5000/api/event-impact');
                console.log("Received event impact data:", response.data);
                setEventData(response.data);
            } catch (error) {
                console.error("Error fetching event impact data:", error);
            }
        };

        const fetchPriceTrends = async () => {
            try {
                const response = await axios.get('http://localhost:5000/api/price-trends');
                const trends = response.data;

                const combinedPrices = [];
                const combinedDates = [];
                const eventDates = []; // To collect event dates

                // Flatten data for the chart
                trends.forEach(trend => {
                    combinedPrices.push(...trend.prices);
                    combinedDates.push(...trend.dates);
                    eventDates.push(trend.date); // Add event date
                });

                // Format dates for chart compatibility
                const formattedDates = combinedDates.map(date => new Date(date).toLocaleDateString());
                
                setPriceTrends({
                    labels: formattedDates,
                    prices: combinedPrices,
                });
                setEventDates(eventDates); // Set event dates
            } catch (error) {
                console.error("Error fetching price trends:", error);
            }
        };

        fetchEventData();
        fetchPriceTrends();
        
    }, []);

    // Prepare data for percentage change charts
    const events = eventData.map(event => event.Event);
    const change1M = eventData.map(event => event.Change_1M || 0); // Default to 0 if null
    const change3M = eventData.map(event => event.Change_3M || 0); // Default to 0 if null
    const change6M = eventData.map(event => event.Change_6M || 0); // Default to 0 if null
    const cumulativeBefore = eventData.map(event => event["Cumulative Return Before"] || 0); // Default to 0 if null
    const cumulativeAfter = eventData.map(event => event["Cumulative Return After"] || 0); // Default to 0 if null

    const percentageChangeData = {
        labels: events,
        datasets: [
            { label: '1 Month', data: change1M, backgroundColor: '#4c6ef5' },
            { label: '3 Months', data: change3M, backgroundColor: '#74c0fc' },
            { label: '6 Months', data: change6M, backgroundColor: '#a5d8ff' }
        ]
    };

    const cumulativeReturnData = {
        labels: events,
        datasets: [
            { label: 'Before Event', data: cumulativeBefore, backgroundColor: '#f59f00' },
            { label: 'After Event', data: cumulativeAfter, backgroundColor: '#ffa94d' }
        ]
    };

    const percentageChangeOptions = {
        scales: {
            x: {
                type: 'category',
                title: {
                    display: true,
                    text: 'Events'
                }
            },
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Percentage Change (%)'
                }
            }
        },
        plugins: {
            legend: { position: 'top' },
            title: { display: true, text: 'Percentage Changes in Oil Prices' }
        }
    };

    const cumulativeReturnOptions = {
        scales: {
            x: {
                type: 'category',
                title: {
                    display: true,
                    text: 'Events'
                }
            },
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Cumulative Return'
                }
            }
        },
        plugins: {
            legend: { position: 'top' },
            title: { display: true, text: 'Cumulative Returns Before and After Events' }
        }
    };

    // Prepare data for price trends chart
    const priceTrendData = {
        labels: priceTrends.labels,
        datasets: [
            {
                label: 'Brent Oil Price',
                data: priceTrends.prices,
                borderColor: '#4c6ef5',
                backgroundColor: 'rgba(76, 110, 245, 0.1)',
                fill: true,
            }
        ]
    };

    const priceTrendOptions = {
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Date'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Price'
                }
            }
        },
        plugins: {
            legend: { position: 'top' },
            title: { display: true, text: 'Brent Oil Price Trends Around Events' },
            annotation: {
                annotations: eventDates.map((event) => ({
                    type: 'line',
                    mode: 'vertical',
                    scaleID: 'x',
                    value: new Date(event).toLocaleDateString(),
                    borderColor: 'red',
                    borderWidth: 2,
                    label: {
                        enabled: true,
                        ontent: event.label || 'Event',
                        position: 'top',
                        backgroundColor: 'rgba(255, 99, 132, 0.8)',
                        color: 'white',
                        content: 'Event',
                        position: 'top'
                    }
                })),
            }
        }
    };

    return (
        <div>
            <h4>Brent Oil Price Trends Around Events</h4>
            <Line data={priceTrendData} options={priceTrendOptions} />

            <h4>Percentage Changes in Oil Prices</h4>
            <Bar data={percentageChangeData} options={percentageChangeOptions} />

            <h4>Cumulative Returns Before and After Events</h4>
            <Bar data={cumulativeReturnData} options={cumulativeReturnOptions} />

            <h4>Statistical Significance (T-Test)</h4>
            <Table striped>
                <thead>
                    <tr>
                        <th>Event</th>
                        <th>Date</th>
                        <th>T-Statistic</th>
                        <th>P-Value</th>
                    </tr>
                </thead>
                <tbody>
                    {eventData.map((event, index) => (
                        <tr key={index}>
                            <td>{event.Event}</td>
                            <td>{event.Date}</td>
                            <td>{event["T-Statistic"] !== null ? event["T-Statistic"].toFixed(2) : 'N/A'}</td>
                            <td>{event["P-Value"] !== null ? event["P-Value"].toFixed(4) : 'N/A'}</td>
                        </tr>
                    ))}
                </tbody>
            </Table>
        </div>
    );
};

export default EventImpact;
