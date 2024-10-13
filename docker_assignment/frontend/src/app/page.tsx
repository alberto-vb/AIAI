"use client";

import { useEffect, useState } from "react";
import { useDraw } from "./hooks/useDraw";

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { Bar, BarChart, CartesianGrid, XAxis } from "recharts";
import { Draw } from "@/app/types/type";

const chartConfig = {
  probability: {
    label: "Probability",
  },
} satisfies ChartConfig;

export default function Home() {
  const { canvasRef, onMouseDown, clear } = useDraw(drawLine);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [chartData, setChartData] = useState([
    { class: "0", probability: 0 },
    { class: "1", probability: 0 },
    { class: "2", probability: 0 },
    { class: "3", probability: 0 },
    { class: "4", probability: 0 },
    { class: "5", probability: 0 },
    { class: "6", probability: 0 },
    { class: "7", probability: 0 },
    { class: "8", probability: 0 },
    { class: "9", probability: 0 },
  ]);

  function drawLine({ prevPoint, currentPoint, ctx }: Draw) {
    const { x: currX, y: currY } = currentPoint;
    console.log();
    const lineColor = "#000";
    const lineWidth = 10;

    const startPoint = prevPoint ?? currentPoint;
    ctx.beginPath();
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = lineColor;
    ctx.moveTo(startPoint.x, startPoint.y);
    ctx.lineTo(currX, currY);
    ctx.stroke();

    ctx.fillStyle = lineColor;
    ctx.beginPath();
    ctx.arc(startPoint.x, startPoint.y, 2, 0, 2 * Math.PI);
    ctx.fill();
  }

  const sendCanvas = async () => {
    const canvas = canvasRef.current;

    if (!canvas) return;

    // Convert the canvas to a data URL (Base64)
    const dataURL = canvas.toDataURL("image/png");

    // Send the Base64 image to the backend
    const response = await fetch("http://localhost:8000/inference", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: dataURL,
      }),
    });

    const result = await response.json();
    console.log(result);
    const updatedChartData = result.probabilities.map(
      (probability: number, index: { toString: () => number }) => ({
        class: index.toString(),
        probability: probability * 100, // Convert to percentage for better readability
      })
    );
    setChartData(updatedChartData);
    setPrediction(result.predicted_digit);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Fill the canvas with a white background on mount
    ctx.fillStyle = "#FFF"; // White background
    ctx.fillRect(0, 0, canvas.width, canvas.height); // Fill canvas with white
  }, []); // Empty dependency array ensures this runs once when the component mounts

  return (
    <div className="flex items-center justify-center h-screen gap-4">
      <div className="flex flex-col items-center justify-center h-screen gap-2">
        <h1>Draw a digit</h1>
        <canvas
          ref={canvasRef}
          width={336}
          height={336}
          style={{
            border: "2px solid black",
          }}
          onMouseDown={onMouseDown}
        />
        <div className="flex gap-2">
          <button
            type="button"
            className="p-2 rounded-md border border-black"
            onClick={() => {
              clear();
              setPrediction(null);
              setChartData([
                { class: "0", probability: 0 },
                { class: "1", probability: 0 },
                { class: "2", probability: 0 },
                { class: "3", probability: 0 },
                { class: "4", probability: 0 },
                { class: "5", probability: 0 },
                { class: "6", probability: 0 },
                { class: "7", probability: 0 },
                { class: "8", probability: 0 },
                { class: "9", probability: 0 },
              ]);
            }}
          >
            Clear canvas
          </button>
          <button
            type="button"
            className="p-2 rounded-md border border-black"
            onClick={sendCanvas}
          >
            Send canvas
          </button>
        </div>
      </div>
      <div className="flex flex-col items-center justify-center h-screen gap-2">
        {prediction !== null && <h1>Predicted digit: {prediction}</h1>}
        {prediction === null && <h1>Digit not predicted yet</h1>}
        <ChartContainer
          config={chartConfig}
          style={{ height: 300, width: 500 }}
        >
          <BarChart accessibilityLayer data={chartData}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="class"
              tickLine={false}
              tickMargin={10}
              axisLine={false}
              tickFormatter={(value) => value.slice(0, 3)}
            />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Bar dataKey="probability" fill="#2563eb" radius={5} />
          </BarChart>
        </ChartContainer>
      </div>
    </div>
  );
}
