"use client";
import React, { useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid } from "recharts"; // Added Legend, CartesianGrid

// --- Constants ---
const API_BASE_URL = "http://127.0.0.1:5001"; // Flask API URL

const CRYPTOS = [
  { symbol: "BTC", name: "Bitcoin" },
  { symbol: "ETH", name: "Ethereum" },
  { symbol: "DOGE", name: "Dogecoin" },
];

const PREDICTION_PERIODS = [
  { value: "day", label: "Next Day" },
  { value: "week", label: "Next Week" },
  { value: "month", label: "Next Month" },
];

const HISTORY_TIMEFRAMES = [
    { value: "week", label: "Last Week" },
    { value: "month", label: "Last Month" },
    { value: "year", label: "Last Year" },
    { value: "all", label: "All Time" },
];

type Action = "history" | "compare" | "predict";

// --- Interfaces ---
interface HistoricalDataPoint {
  date: string; // Expecting 'YYYY-MM-DD' string from API
  price: number;
}

interface ComparisonDataPoint {
  date: string; // Expecting 'YYYY-MM-DD' string from API
  actual: number;
  predicted: number;
}

interface PredictionPoint {
  date: string; // Expecting 'YYYY-MM-DD' string from API
  prediction: number;
}

// --- Helper Functions ---
// Formatter for Y-axis (optional, for better readability)
const formatPrice = (tickItem: number) => {
    return tickItem.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

// --- Component ---
export default function Home() {
  // State variables
  const [selectedAction, setSelectedAction] = useState<Action>("history");
  const [selectedCrypto, setSelectedCrypto] = useState("BTC");
  const [selectedPredictionPeriod, setSelectedPredictionPeriod] = useState("day");
  const [selectedTimeframe, setSelectedTimeframe] = useState("week"); // For history/compare

  const [historicalData, setHistoricalData] = useState<HistoricalDataPoint[] | null>(null);
  const [comparisonData, setComparisonData] = useState<ComparisonDataPoint[] | null>(null);
  const [predictionData, setPredictionData] = useState<PredictionPoint[] | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // --- Data Fetching Functions ---
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    setHistoricalData(null);
    setComparisonData(null);
    setPredictionData(null);

    let url = "";
    let params = new URLSearchParams();
    params.append("symbol", selectedCrypto);

    try {
      if (selectedAction === "history") {
        url = `${API_BASE_URL}/historical`;
        params.append("time_frame", selectedTimeframe);
      } else if (selectedAction === "compare") {
        url = `${API_BASE_URL}/compare`;
        params.append("time_frame", selectedTimeframe);
      } else if (selectedAction === "predict") {
        url = `${API_BASE_URL}/predict`;
        params.append("period", selectedPredictionPeriod);
      } else {
        throw new Error("Invalid action selected");
      }

      const fullUrl = `${url}?${params.toString()}`;
      console.log(`Fetching data from: ${fullUrl}`); // Log the URL being fetched
      const res = await fetch(fullUrl);

      if (!res.ok) {
        let errorData;
        try {
          errorData = await res.json();
        } catch {
          const text = await res.text();
          errorData = { error: text || `API request failed with status: ${res.status}` };
        }
        console.error("API Error Response:", errorData);
        throw new Error(errorData.error || `API Error: ${res.status}`);
      }

      const json = await res.json();
      console.log("API Success Response:", json); // Log successful response

      if (selectedAction === "history") {
        if (json.data && Array.isArray(json.data)) {
          setHistoricalData(json.data);
        } else {
          throw new Error("Invalid historical data format received");
        }
      } else if (selectedAction === "compare") {
        if (json.data && Array.isArray(json.data)) {
          setComparisonData(json.data);
        } else {
          throw new Error("Invalid comparison data format received");
        }
      } else if (selectedAction === "predict") {
        if (json.predictions && json.dates && Array.isArray(json.predictions) && Array.isArray(json.dates)) {
           setPredictionData(
             json.dates.map((date: string, i: number) => ({
               date,
               prediction: json.predictions[i],
             }))
           );
        } else {
          throw new Error("Invalid prediction data format received");
        }
      }
    } catch (e: any) {
      console.error("Fetch Error:", e);
      setError(e.message || "Failed to fetch data");
    } finally {
      setLoading(false);
    }
  };

  // --- Render Logic ---
  const renderChart = () => {
    if (loading) return <div className="flex justify-center items-center h-64"><p>Loading chart data...</p></div>;
    if (error) return <div className="text-red-500 text-center h-64 flex items-center justify-center"><p>Error loading data: {error}</p></div>;

    let chartData: any[] = [];
    let lines: React.ReactElement[] = []; // Use React.ReactElement for JSX elements
    let title = "";
    const cryptoName = CRYPTOS.find(c => c.symbol === selectedCrypto)?.name || selectedCrypto;

    if (selectedAction === "history" && historicalData) {
        chartData = historicalData;
        lines.push(<Line key="price" type="monotone" dataKey="price" name="Price" stroke="#2563eb" dot={false} strokeWidth={2} isAnimationActive={false} />);
        const timeframeLabel = HISTORY_TIMEFRAMES.find(t => t.value === selectedTimeframe)?.label || selectedTimeframe;
        title = `${cryptoName} Historical Price (${timeframeLabel})`;
    } else if (selectedAction === "compare" && comparisonData) {
        chartData = comparisonData;
        lines.push(<Line key="actual" type="monotone" dataKey="actual" name="Actual Price" stroke="#2563eb" dot={false} strokeWidth={2} isAnimationActive={false} />);
        lines.push(<Line key="predicted" type="monotone" dataKey="predicted" name="Predicted Price" stroke="#16a34a" dot={false} strokeWidth={2} isAnimationActive={false} />);
        const timeframeLabel = HISTORY_TIMEFRAMES.find(t => t.value === selectedTimeframe)?.label || selectedTimeframe;
        title = `${cryptoName} Actual vs. Predicted (${timeframeLabel})`;
    } else if (selectedAction === "predict" && predictionData) {
        chartData = predictionData;
        lines.push(<Line key="prediction" type="monotone" dataKey="prediction" name="Predicted Price" stroke="#16a34a" dot={true} strokeWidth={2} isAnimationActive={false} />);
        const periodLabel = PREDICTION_PERIODS.find(p => p.value === selectedPredictionPeriod)?.label || selectedPredictionPeriod;
        title = `${cryptoName} Price Prediction (${periodLabel})`;
    } else {
        return <div className="flex justify-center items-center h-64"><p>Select options and click 'Get Data' to view the chart.</p></div>;
    }

    return (
      <>
        <h3 className="text-lg font-semibold mb-2 text-center text-gray-700 dark:text-gray-200">{title}</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ccc" /> {/* Use a neutral stroke color */}
            <XAxis dataKey="date" tick={{ fontSize: 12 }} />
            <YAxis tickFormatter={formatPrice} domain={["auto", "auto"]} tick={{ fontSize: 12 }} width={80} />
            <Tooltip formatter={(value: number) => formatPrice(value)} />
            <Legend />
            {lines}
          </LineChart>
        </ResponsiveContainer>
      </>
    );
  };

  return (
    <div className="flex flex-col items-center justify-start min-h-screen bg-gray-100 dark:bg-gray-900 p-4 sm:p-8">
      <h1 className="text-3xl font-bold mb-6 text-center text-gray-800 dark:text-gray-100">
        Crypto Price Analysis & Prediction
      </h1>

      {/* --- Controls Section --- */}
      <div className="w-full max-w-4xl bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          {/* Action Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Select Action</label>
            <select
              className="w-full p-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100 focus:ring-blue-500 focus:border-blue-500"
              value={selectedAction}
              onChange={(e) => setSelectedAction(e.target.value as Action)}
            >
              <option value="history">View Historical Data</option>
              <option value="compare">Compare Actual vs. Predicted</option>
              <option value="predict">Predict Future Prices</option>
            </select>
          </div>

          {/* Crypto Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Select Crypto</label>
            <select
              className="w-full p-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100 focus:ring-blue-500 focus:border-blue-500"
              value={selectedCrypto}
              onChange={(e) => setSelectedCrypto(e.target.value)}
            >
              {CRYPTOS.map((c) => (
                <option key={c.symbol} value={c.symbol}>
                  {c.name} ({c.symbol})
                </option>
              ))}
            </select>
          </div>

          {/* Timeframe/Period Selection (Conditional) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              {selectedAction === 'predict' ? 'Select Prediction Period' : 'Select Time Frame'}
            </label>
            {selectedAction === 'predict' ? (
              <select
                className="w-full p-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100 focus:ring-blue-500 focus:border-blue-500"
                value={selectedPredictionPeriod}
                onChange={(e) => setSelectedPredictionPeriod(e.target.value)}
              >
                {PREDICTION_PERIODS.map((p) => (
                  <option key={p.value} value={p.value}>
                    {p.label}
                  </option>
                ))}
              </select>
            ) : (
              <select
                className="w-full p-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100 focus:ring-blue-500 focus:border-blue-500"
                value={selectedTimeframe}
                onChange={(e) => setSelectedTimeframe(e.target.value)}
              >
                {HISTORY_TIMEFRAMES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>

        {/* Fetch Button and Error Message */}
        <div className="flex items-center justify-center mt-4">
           <button
             className="px-6 py-2 rounded bg-blue-600 text-white font-semibold hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
             onClick={fetchData}
             disabled={loading}
           >
             {loading ? "Loading..." : "Get Data"}
           </button>
        </div>
         {error && !loading && <div className="text-red-500 mt-4 text-center">{error}</div>}
      </div>

      {/* --- Chart Section --- */}
      <div className="w-full max-w-4xl bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
        {renderChart()}
      </div>

       <p className="mt-8 text-gray-500 dark:text-gray-400 text-sm text-center">
         Powered by your local prediction API.
       </p>
    </div>
  );
}
