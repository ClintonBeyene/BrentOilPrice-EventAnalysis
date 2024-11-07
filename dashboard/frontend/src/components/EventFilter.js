// frontend/src/components/EventFilter.js

import React, { useState } from 'react';
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";
import { Button } from 'react-bootstrap';

const EventFilter = ({ onFilter }) => {
    const [startDate, setStartDate] = useState(null);
    const [endDate, setEndDate] = useState(null);

    const handleFilter = () => {
        // Call the onFilter function with the selected dates
        onFilter(startDate, endDate);
    };

    return (
        <div className="mb-3">
            <h3>Filter Events</h3>
            <DatePicker 
                selected={startDate} 
                onChange={date => setStartDate(date)} 
                placeholderText="Start Date" 
                dateFormat="yyyy/MM/dd"
            />
            <DatePicker 
                selected={endDate} 
                onChange={date => setEndDate(date)} 
                placeholderText="End Date" 
                dateFormat="yyyy/MM/dd"
            />
            <Button variant="primary" onClick={handleFilter}>Apply Filter</Button>
        </div>
    );
}

export default EventFilter;