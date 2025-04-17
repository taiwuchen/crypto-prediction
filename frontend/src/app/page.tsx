"use client";
import { useEffect, useRef, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

// Example: Binance BTC/USDT WebSocket endpoint
const WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade";

interface PricePoint {
  time: string;
  price: number;
}

export default function Home() {
  const [data, setData] = useState<PricePoint[]>([]);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket(WS_URL);
    ws.current.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const price = parseFloat(msg.p);
      const time = new Date(msg.T).toLocaleTimeString();
      setData((prev) => [
        ...prev.slice(-49), // keep last 49 points
        { time, price },
      ]);
    };
    return () => {
      ws.current?.close();
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 dark:bg-gray-900 p-8">
      <h1 className="text-2xl font-bold mb-4 text-center text-gray-800 dark:text-gray-100">
        Real-Time BTC/USDT Price (Binance Demo)
      </h1>
      <div className="w-full max-w-2xl h-80 bg-white dark:bg-gray-800 rounded shadow p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <XAxis dataKey="time" minTickGap={20} tick={{ fontSize: 12 }} />
            <YAxis domain={["auto", "auto"]} tick={{ fontSize: 12 }} />
            <Tooltip />
            <Line type="monotone" dataKey="price" stroke="#2563eb" dot={false} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-4 text-gray-500 text-sm text-center">
        Powered by Binance public WebSocket. This is a demo for real-time charting.
      </p>
    </div>
  );
}
