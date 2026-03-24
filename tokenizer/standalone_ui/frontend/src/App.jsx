import React, { useState, useEffect } from 'react';
import { Network, Cpu, Activity, FileJson, Play, Pause, RotateCcw, Database, CheckCircle, Loader2 } from 'lucide-react';

export default function App() {
  const [step, setStep] = useState(0); 
  const [isPlaying, setIsPlaying] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [outputType, setOutputType] = useState('token'); 
  const [visibleTokens, setVisibleTokens] = useState(0);
  const [hoveredEdge, setHoveredEdge] = useState(null);

  // Dynamic State Variables!
  const [prompt, setPrompt] = useState("What is the capital of Australia?");
  const [tokens, setTokens] = useState(["Waiting", "for", "backend..."]);
  const [tokenScores, setTokenScores] = useState([]);
  const [responseScore, setResponseScore] = useState(0.0);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);

  // Orchestrate the animation sequence
  useEffect(() => {
    let timer;
    if (isPlaying) {
      if (step === 0) setStep(1);
      else if (step === 1) {
        if (visibleTokens < tokens.length) {
          timer = setTimeout(() => setVisibleTokens(v => v + 1), 400);
        } else {
          timer = setTimeout(() => setStep(2), 1500);
        }
      } else if (step === 2) {
        timer = setTimeout(() => setStep(3), 2500);
      } else if (step === 3) {
        timer = setTimeout(() => setStep(4), 4000);
      } else if (step === 4) {
        setIsPlaying(false);
      }
    }
    return () => clearTimeout(timer);
  }, [isPlaying, step, visibleTokens, tokens.length]);

  const reset = () => {
    setIsPlaying(false);
    setStep(0);
    setVisibleTokens(0);
  };

  // --- NEW: The Live API Fetch ---
  const handleAnalyze = async () => {
    if (!prompt.trim()) return;
    reset();
    setIsFetching(true);
    
    try {
      const res = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: prompt })
      });
      
      const data = await res.json();
      
      if (data.error) {
        alert("Backend Error: " + data.error);
        setIsFetching(false);
        return;
      }

      // Overwrite state with real GNN data
      setTokens(data.tokens);
      setNodes(data.nodes);
      setEdges(data.edges);
      setTokenScores(data.tokenScores);
      setResponseScore(data.responseScore);
      
      setIsFetching(false);
      setIsPlaying(true); // Start the show!

    } catch (error) {
      alert("Make sure your Python API is running on Port 8000! Error: " + error.message);
      setIsFetching(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6 font-sans text-slate-800 flex flex-col items-center">
      <style>{`
        @keyframes flow { to { stroke-dashoffset: -20; } }
        .animate-flow { stroke-dasharray: 6 6; animation: flow 0.8s linear infinite; }
        @keyframes pulse-node {
          0%, 100% { transform: scale(1); filter: drop-shadow(0 0 0px rgba(99, 102, 241, 0)); }
          50% { transform: scale(1.05); filter: drop-shadow(0 0 8px rgba(99, 102, 241, 0.6)); }
        }
        .node-pulse { animation: pulse-node 1.5s ease-in-out infinite; transform-origin: center; }
      `}</style>

      <div className="max-w-5xl w-full space-y-6">
        
        {/* Header & Controls */}
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col gap-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-3">
                <Network className="text-indigo-600" />
                CHARM Live Inference
              </h1>
              <p className="text-slate-500 mt-1 text-sm">GNN Hallucination Detection Engine</p>
            </div>
            
            <div className="flex items-center gap-3">
              <div className="bg-slate-100 p-1 rounded-lg flex mr-4">
                <button onClick={() => setOutputType('token')} className={`px-3 py-1 text-sm font-medium rounded ${outputType === 'token' ? 'bg-white shadow-sm text-indigo-600' : 'text-slate-500'}`}>Token Level</button>
                <button onClick={() => setOutputType('response')} className={`px-3 py-1 text-sm font-medium rounded ${outputType === 'response' ? 'bg-white shadow-sm text-indigo-600' : 'text-slate-500'}`}>Response Level</button>
              </div>
            </div>
          </div>

          {/* Prompt Input Bar */}
          <div className="flex gap-3 mt-2">
            <input 
              type="text" 
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Ask the model a question to analyze..."
              className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              disabled={isFetching || isPlaying}
            />
            <button 
              onClick={handleAnalyze} 
              disabled={isFetching || isPlaying}
              className="flex items-center gap-2 bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 transition-colors font-medium disabled:opacity-50"
            >
              {isFetching ? <Loader2 size={18} className="animate-spin" /> : <Play size={18} />}
              {isFetching ? 'Analyzing...' : 'Generate & Analyze'}
            </button>
            <button onClick={reset} className="p-2 text-slate-400 hover:text-slate-600 bg-slate-100 rounded-lg">
              <RotateCcw size={18} />
            </button>
          </div>
        </div>

        {/* Stepper */}
        <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200">
          <div className="flex justify-between items-center relative">
            <div className="absolute top-1/2 left-0 w-full h-1 bg-slate-100 -z-10 -translate-y-1/2 rounded"></div>
            <div className={`absolute top-1/2 left-0 h-1 bg-indigo-500 -z-10 -translate-y-1/2 rounded transition-all duration-700`} style={{ width: `${(step / 4) * 100}%` }}></div>
            {[
              { num: 1, title: "LLM Generation", icon: Cpu },
              { num: 2, title: "Graph Construction", icon: Database },
              { num: 3, title: "Message Passing", icon: Activity },
              { num: 4, title: "Detection Output", icon: FileJson }
            ].map((s, i) => {
              const isActive = step >= s.num;
              return (
                <div key={i} className={`flex flex-col items-center gap-2 transition-opacity ${isActive ? 'opacity-100' : 'opacity-40'}`}>
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 ${isActive ? 'bg-indigo-600 border-indigo-600 text-white' : 'bg-white border-slate-300 text-slate-400'}`}>
                    <s.icon size={18} />
                  </div>
                  <span className={`text-xs font-semibold ${isActive ? 'text-indigo-900' : 'text-slate-500'}`}>{s.title}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* The Animated Theater */}
        <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200 min-h-[500px] flex flex-col justify-center relative overflow-hidden">
          
          {step === 0 && !isFetching && (
            <div className="text-center text-slate-400">
              <Network size={48} className="mx-auto mb-4 opacity-30" />
              <p>Type a prompt above and click "Generate & Analyze"</p>
            </div>
          )}

          {isFetching && (
            <div className="text-center text-indigo-500 animate-pulse">
              <Loader2 size={48} className="mx-auto mb-4 animate-spin opacity-50" />
              <p>LLM is generating and GNN is scoring...</p>
            </div>
          )}

          {/* 1. Tokens & Extraction (Now stays visible for context) */}
          {nodes.length > 0 && (
            <div className={`absolute top-8 left-8 transition-all duration-1000 
              ${step >= 2 ? 'opacity-20 scale-90 -translate-y-10 pointer-events-none' : 'opacity-100'}`}>
              <div className="font-mono text-sm text-slate-500 mb-2">PROMPT: {prompt}</div>
              <div className="text-xl font-medium flex gap-1 flex-wrap mb-8">
                {tokens.map((t, i) => (
                  <span key={i} className={`transition-all duration-300 ${i < visibleTokens ? 'opacity-100' : 'opacity-0 translate-y-4'}`}>
                    {t}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* 2. SVG Graph (Order matters: Edges first, then Nodes) */}
          {step >= 2 && nodes.length > 0 && (
            <div className="w-full h-[450px] relative animate-fade-in">
              <svg width="100%" height="100%" viewBox="0 0 850 400" className="overflow-visible">
                <defs>
                  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="28" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill={step === 3 ? "#6366f1" : "#cbd5e1"} />
                  </marker>
                </defs>

                {/* RENDER EDGES FIRST */}
                {edges.map((edge, i) => {
                  const s = nodes.find(n => n.id === edge.s);
                  const t = nodes.find(n => n.id === edge.t);
                  if (!s || !t) return null;
                  const cx = (s.x + t.x) / 2;
                  const cy = Math.min(s.y, t.y) - 60 + ((i % 5) * 10);
                  const d = `M ${s.x} ${s.y} Q ${cx} ${cy} ${t.x} ${t.y}`;
                  const isHovered = hoveredEdge === i;

                  return (
                    <g key={`edge-${i}`}>
                      {/* Invisible Hitbox for Hovers */}
                      <path 
                        d={d} fill="none" stroke="transparent" strokeWidth="20" className="cursor-pointer"
                        onMouseEnter={() => setHoveredEdge(i)}
                        onMouseLeave={() => setHoveredEdge(null)}
                      />
                      {/* Static Edge */}
                      <path 
                        d={d} fill="none"
                        stroke={isHovered ? "#4f46e5" : "#e2e8f0"}
                        strokeWidth={isHovered ? "3" : "1.5"}
                        markerEnd="url(#arrowhead)"
                      />
                      {/* Purple Message Flow (Step 3) */}
                      {step === 3 && <path d={d} fill="none" stroke="#818cf8" strokeWidth="2.5" className="animate-flow pointer-events-none" />}
                      {/* Tooltip for the Edge Weight */}
                      {isHovered && (
                        <g transform={`translate(${cx}, ${cy - 15})`} className="pointer-events-none">
                          <rect x="-22" y="-12" width="44" height="20" rx="4" fill="#4f46e5" />
                          <text textAnchor="middle" y="2" fontSize="11" fill="white" fontWeight="bold">{edge.weight}</text>
                        </g>
                      )}
                    </g>
                  );
                })}

                {/* RENDER NODES LAST (So they are on top) */}
                {nodes.map((node, i) => {
                  const score = tokenScores[i] || 0;
                  const isHallucination = score > 0.4; // Threshold for color
                  const showScore = step === 4;
                  
                  return (
                    <g
                      key={`node-${node.id}`}
                      transform={`translate(${node.x}, ${node.y})`}
                      className={`transition-all duration-700 ${step === 3 ? 'node-pulse' : ''}`}
                    >
                      <circle 
                        r="28"
                        fill={showScore ? (isHallucination ? "#fee2e2" : "#f0fdf4") : "white"}
                        stroke={showScore ? (isHallucination ? "#ef4444" : "#22c55e") : (step === 3 ? "#6366f1" : "#e2e8f0")}
                        strokeWidth={showScore ? "3" : "2"}
                        className="transition-colors duration-500 shadow-sm"
                      />
                      <text textAnchor="middle" y="5" fontSize="11" fontWeight="600" className="fill-slate-700 pointer-events-none">
                        {node.label}
                      </text>
                      {showScore && (
                        <g transform="translate(0, -38)">
                          <rect x="-18" y="-10" width="36" height="18" rx="4" fill={isHallucination ? "#ef4444" : "#22c55e"} />
                          <text textAnchor="middle" y="3" fontSize="10" fill="white" fontWeight="bold">{score.toFixed(2)}</text>
                        </g>
                      )}
                    </g>
                  );
                })}
              </svg>

              {/* Response Score Dashboard */}
              {step === 4 && outputType === 'response' && (
                <div className="absolute bottom-0 left-1/2 -translate-x-1/2 flex flex-col items-center bg-white p-4 rounded-xl shadow-lg border border-slate-200 z-10 w-64">
                  <div className="text-xs font-bold text-slate-500 uppercase mb-2">Global Hallucination Score</div>
                  <div className="text-3xl font-bold text-slate-800">{responseScore.toFixed(2)}</div>
                  <div className={`text-xs font-medium mt-1 ${responseScore > 0.5 ? 'text-red-600' : 'text-green-600'}`}>
                    {responseScore > 0.5 ? 'High probability of hallucination.' : 'Response appears structurally faithful.'}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}