import { useState, useRef } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileRef = useRef(null);

  const handleFile = (file) => {
    if (file && file.type.startsWith("image/")) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === "dragenter" || e.type === "dragover");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  const handlePredict = async () => {
    if (!image) return;
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });
      setResult(await res.json());
    } catch {
      setResult({
        status: "error",
        message: "Backend not running. Start Flask: python app.py",
      });
    }
    setLoading(false);
  };

  const handleReset = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  const confLevel = (c) => (c >= 0.8 ? "high" : c >= 0.5 ? "mid" : "low");

  return (
    <div className="app">
      <div className="container">
        {/* ========== LEFT ========== */}
        <div className="left-panel">
          <div className="header">
            <span className="header-icon">🌿</span>
            <h1>Crop Disease Detector</h1>
            <p>Supports Pepper · Potato · Tomato</p>
          </div>

          <div
            className={`upload-area ${dragActive ? "drag-active" : ""}`}
            onClick={() => fileRef.current?.click()}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
          >
            <span className="upload-icon">📤</span>
            <p>Drag & drop your leaf image</p>
            <p className="browse">or click to browse</p>
            {image && <p className="filename">📄 {image.name}</p>}
            <input
              type="file"
              ref={fileRef}
              onChange={(e) => handleFile(e.target.files[0])}
              accept="image/*"
              style={{ display: "none" }}
            />
          </div>

          {preview && (
            <div className="preview-box">
              <img src={preview} alt="preview" />
            </div>
          )}
        </div>

        {/* ========== RIGHT ========== */}
        <div className="right-panel">
          <div className="right-title">🔬 Analysis Results</div>

          {/* Buttons */}
          <div className="btn-group">
            <button
              className="predict-btn"
              onClick={handlePredict}
              disabled={!image || loading}
            >
              {loading ? "⏳ Analyzing..." : "🔍 Analyze"}
            </button>
            {(preview || result) && (
              <button className="reset-btn" onClick={handleReset}>
                🔄 Reset
              </button>
            )}
          </div>

          {/* Loader */}
          {loading && (
            <div className="loader-box">
              <div className="spinner"></div>
              <span>Analyzing your image...</span>
            </div>
          )}

          {/* Results Area */}
          <div className="result-area">
            {/* SUCCESS - Healthy */}
            {result?.status === "success" && result.is_healthy && (
              <div className="result-card healthy">
                <span className="result-icon">✅</span>
                <div className="result-title">Healthy Plant!</div>

                <div className="info-row">
                  <span className="icon">🌱</span>
                  <span className="label">Plant:</span>
                  <span className="value">{result.plant}</span>
                </div>
                <div className="info-row">
                  <span className="icon">🔬</span>
                  <span className="label">Status:</span>
                  <span className="value">{result.disease}</span>
                </div>

                <div className="conf-section">
                  <div className="conf-header">
                    <span className="conf-label">📊 Confidence</span>
                    <span className={`conf-val ${confLevel(result.confidence)}`}>
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="conf-bar-bg">
                    <div
                      className={`conf-bar ${confLevel(result.confidence)}`}
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>

                {result.remedy && (
                  <div className="remedy-box">
                    <h4>🌟 Care Tips</h4>
                    <p>{result.remedy}</p>
                  </div>
                )}
              </div>
            )}

            {/* SUCCESS - Disease */}
            {result?.status === "success" && !result.is_healthy && (
              <div className="result-card diseased">
                <span className="result-icon">⚠️</span>
                <div className="result-title">Disease Detected</div>

                <div className="info-row">
                  <span className="icon">🌱</span>
                  <span className="label">Plant:</span>
                  <span className="value">{result.plant}</span>
                </div>
                <div className="info-row">
                  <span className="icon">🦠</span>
                  <span className="label">Disease:</span>
                  <span className="value">{result.disease}</span>
                </div>

                <div className="conf-section">
                  <div className="conf-header">
                    <span className="conf-label">📊 Confidence</span>
                    <span className={`conf-val ${confLevel(result.confidence)}`}>
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="conf-bar-bg">
                    <div
                      className={`conf-bar ${confLevel(result.confidence)}`}
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>

                {result.remedy && (
                  <div className="remedy-box">
                    <h4>💊 Recommended Treatment</h4>
                    <p>{result.remedy}</p>
                  </div>
                )}
              </div>
            )}

            {/* NON LEAF / UNSUPPORTED */}
            {(result?.status === "non_leaf" ||
              result?.status === "unsupported" ||
              result?.status === "unknown") && (
              <div className="result-card rejected">
                <span className="result-icon">🚫</span>
                <div className="result-title">Not Recognized</div>
                <p className="rejected-msg">{result.message}</p>
                <div className="supported-list">
                  <div className="supported-item">🫑 Pepper</div>
                  <div className="supported-item">🥔 Potato</div>
                  <div className="supported-item">🍅 Tomato</div>
                </div>
              </div>
            )}

            {/* ERROR */}
            {result?.status === "error" && (
              <div className="result-card error">
                <span className="result-icon">❌</span>
                <div className="result-title">Error</div>
                <p className="rejected-msg">{result.message}</p>
              </div>
            )}

            {/* PLACEHOLDER */}
            {!result && !loading && (
              <div className="placeholder">
                <span className="placeholder-icon">🍃</span>
                <p>Upload a plant leaf image and click Analyze to detect diseases</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;