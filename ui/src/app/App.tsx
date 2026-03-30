import { useEffect, useMemo, useState } from 'react';
import { VideoInput } from './components/VideoInput';
import { ProcessingPipeline } from './components/ProcessingPipeline';
import { KeypointsDisplay } from './components/KeypointsDisplay';
import { TranslationOutput } from './components/TranslationOutput';
import { Sidebar, type SidebarSection } from './components/Sidebar';

type PredictionResult = {
  label: string;
  score: number;
  elapsed_ms: number;
  uid?: string;
};

type ResultRecord = PredictionResult & {
  id: string;
  source: 'upload' | 'camera';
  filename: string;
  created_at: string;
};

const RECORDING_MS = 2000;
const MAX_HISTORY = 200;
const HISTORY_STORAGE_KEY = 'lumisign_history_v1';
const SAVED_STORAGE_KEY = 'lumisign_saved_v1';

function createRecordId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function loadStoredRecords(key: string): ResultRecord[] {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((entry): entry is ResultRecord => {
      return (
        entry !== null &&
        typeof entry === 'object' &&
        typeof (entry as { id?: unknown }).id === 'string' &&
        typeof (entry as { label?: unknown }).label === 'string' &&
        typeof (entry as { score?: unknown }).score === 'number' &&
        typeof (entry as { elapsed_ms?: unknown }).elapsed_ms === 'number'
      );
    });
  } catch {
    return [];
  }
}

function formatDateTime(iso: string) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

function escapeCsvValue(value: string | number | boolean) {
  const text = String(value);
  if (text.includes('"') || text.includes(',') || text.includes('\n')) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

export default function App() {
  const [activeSection, setActiveSection] = useState<SidebarSection>('new');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [history, setHistory] = useState<ResultRecord[]>(() => loadStoredRecords(HISTORY_STORAGE_KEY));
  const [saved, setSaved] = useState<ResultRecord[]>(() => loadStoredRecords(SAVED_STORAGE_KEY));
  const [latestRecord, setLatestRecord] = useState<ResultRecord | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (!videoFile) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(videoFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  useEffect(() => {
    return () => {
      cameraStream?.getTracks().forEach((track) => track.stop());
    };
  }, [cameraStream]);

  useEffect(() => {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
  }, [history]);

  useEffect(() => {
    localStorage.setItem(SAVED_STORAGE_KEY, JSON.stringify(saved));
  }, [saved]);

  useEffect(() => {
    if (!notice) return;
    const timeout = window.setTimeout(() => setNotice(null), 3000);
    return () => window.clearTimeout(timeout);
  }, [notice]);

  const stopCamera = () => {
    cameraStream?.getTracks().forEach((track) => track.stop());
    setCameraStream(null);
  };

  const handleToggleCamera = async () => {
    if (cameraStream) {
      stopCamera();
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      setCameraStream(stream);
      setActiveSection('new');
      setVideoFile(null);
      setPrediction(null);
      setError(null);
      setCurrentStep(0);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unable to access camera.';
      setError(message);
    }
  };

  const handleVideoUpload = (file: File) => {
    if (cameraStream) {
      stopCamera();
    }
    setActiveSection('new');
    setVideoFile(file);
    setPrediction(null);
    setError(null);
    setCurrentStep(0);
  };

  const recordClip = (stream: MediaStream, durationMs: number) => {
    return new Promise<Blob>((resolve, reject) => {
      const options: MediaRecorderOptions = {};
      if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) {
        options.mimeType = 'video/webm;codecs=vp9';
      } else if (MediaRecorder.isTypeSupported('video/webm;codecs=vp8')) {
        options.mimeType = 'video/webm;codecs=vp8';
      } else if (MediaRecorder.isTypeSupported('video/webm')) {
        options.mimeType = 'video/webm';
      }

      const recorder = new MediaRecorder(stream, options);
      const chunks: BlobPart[] = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };
      recorder.onerror = () => reject(new Error('Recording failed.'));
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: recorder.mimeType || 'video/webm' });
        resolve(blob);
      };

      recorder.start();
      window.setTimeout(() => {
        if (recorder.state !== 'inactive') {
          recorder.stop();
        }
      }, durationMs);
    });
  };

  const requestPrediction = async (blob: Blob, filename: string) => {
    const form = new FormData();
    form.append('file', blob, filename);
    const resp = await fetch('/predict', { method: 'POST', body: form });
    const responseText = await resp.text();
    let data: unknown = null;
    if (responseText) {
      try {
        data = JSON.parse(responseText);
      } catch {
        data = null;
      }
    }

    if (!resp.ok) {
      const errorMessage =
        data && typeof data === 'object' && 'error' in data
          ? String((data as { error?: unknown }).error ?? '')
          : responseText;
      throw new Error(errorMessage || `Request failed (${resp.status})`);
    }

    if (!data || typeof data !== 'object') {
      throw new Error('Backend returned an empty response.');
    }

    return data as PredictionResult;
  };

  const handleCopyPrediction = async () => {
    const record = latestRecord;
    if (!record) {
      setNotice('No prediction available to copy.');
      return;
    }
    const text = [
      `Label: ${record.label}`,
      `Confidence: ${(record.score * 100).toFixed(1)}%`,
      `Processing Time: ${record.elapsed_ms} ms`,
      `Source: ${record.source}`,
      `Timestamp: ${formatDateTime(record.created_at)}`,
    ].join('\n');
    try {
      await navigator.clipboard.writeText(text);
      setNotice('Prediction copied to clipboard.');
    } catch {
      setNotice('Failed to copy. Please allow clipboard access.');
    }
  };

  const handleSaveCurrentResult = () => {
    if (!latestRecord) {
      setNotice('No prediction available to save.');
      return;
    }
    setSaved((prev) => {
      if (prev.some((item) => item.id === latestRecord.id)) {
        return prev;
      }
      return [latestRecord, ...prev];
    });
    setNotice('Result saved.');
  };

  const hasSavedCurrent = latestRecord ? saved.some((item) => item.id === latestRecord.id) : false;

  const handleExportResults = () => {
    const merged = new Map<string, ResultRecord & { saved: boolean }>();
    for (const item of history) {
      merged.set(item.id, { ...item, saved: false });
    }
    for (const item of saved) {
      const existing = merged.get(item.id);
      if (existing) {
        merged.set(item.id, { ...existing, saved: true });
      } else {
        merged.set(item.id, { ...item, saved: true });
      }
    }

    const rows = Array.from(merged.values()).sort((a, b) =>
      b.created_at.localeCompare(a.created_at),
    );

    if (rows.length === 0) {
      setNotice('No results to export yet.');
      return;
    }

    const headers = ['id', 'label', 'confidence', 'elapsed_ms', 'source', 'filename', 'created_at', 'saved'];
    const csvLines = [
      headers.join(','),
      ...rows.map((item) =>
        [
          item.id,
          item.label,
          item.score.toFixed(6),
          item.elapsed_ms,
          item.source,
          item.filename,
          item.created_at,
          item.saved,
        ]
          .map(escapeCsvValue)
          .join(','),
      ),
    ];

    const blob = new Blob([csvLines.join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    link.href = url;
    link.download = `lumisign-results-${stamp}.csv`;
    link.click();
    URL.revokeObjectURL(url);
    setNotice('Export complete.');
  };

  const handleStartTranslation = async () => {
    if (isProcessing) return;
    if (!videoFile && !cameraStream) {
      setError('Please upload a video or enable the camera.');
      return;
    }

    setIsProcessing(true);
    setCurrentStep(1);
    setPrediction(null);
    setError(null);

    try {
      let blob: Blob;
      let filename: string;
      let source: ResultRecord['source'];

      if (videoFile) {
        blob = videoFile;
        filename = videoFile.name;
        source = 'upload';
      } else if (cameraStream) {
        setCurrentStep(2);
        blob = await recordClip(cameraStream, RECORDING_MS);
        filename = 'recording.webm';
        source = 'camera';
      } else {
        setIsProcessing(false);
        return;
      }

      setCurrentStep(3);
      const result = await requestPrediction(blob, filename);
      setPrediction(result);
      const record: ResultRecord = {
        ...result,
        id: createRecordId(),
        source,
        filename,
        created_at: new Date().toISOString(),
      };
      setLatestRecord(record);
      setHistory((prev) => [record, ...prev].slice(0, MAX_HISTORY));
      setCurrentStep(4);
      setActiveSection('new');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Prediction failed.';
      setError(message);
      setCurrentStep(0);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setIsProcessing(false);
    setCurrentStep(0);
    setVideoFile(null);
    setPrediction(null);
    setError(null);
    stopCamera();
  };

  const analytics = useMemo(() => {
    const total = history.length;
    const averageConfidence =
      total > 0 ? history.reduce((sum, item) => sum + item.score, 0) / total : 0;
    const labelCounts = new Map<string, number>();
    for (const item of history) {
      labelCounts.set(item.label, (labelCounts.get(item.label) ?? 0) + 1);
    }
    const topLabels = Array.from(labelCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    return {
      total,
      saved: saved.length,
      averageConfidence,
      topLabels,
    };
  }, [history, saved.length]);

  const renderRecordList = (records: ResultRecord[], emptyText: string) => {
    if (records.length === 0) {
      return <div className="text-sm text-gray-500">{emptyText}</div>;
    }
    return (
      <div className="space-y-2">
        {records.slice(0, 20).map((item) => (
          <button
            key={item.id}
            onClick={() => {
              setPrediction(item);
              setLatestRecord(item);
              setCurrentStep(4);
              setActiveSection('new');
            }}
            className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-left hover:bg-gray-50"
          >
            <div className="font-medium text-sm">{item.label}</div>
            <div className="text-xs text-gray-500">
              {(item.score * 100).toFixed(1)}% confidence • {item.elapsed_ms} ms •{' '}
              {formatDateTime(item.created_at)}
            </div>
          </button>
        ))}
      </div>
    );
  };

  const renderActiveSection = () => {
    switch (activeSection) {
      case 'history':
        return (
          <div className="mb-6 rounded-lg border border-gray-200 bg-white p-6">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg">History</h2>
              <button
                onClick={() => setHistory([])}
                className="rounded border border-gray-300 px-3 py-1.5 text-xs hover:bg-gray-50"
              >
                Clear History
              </button>
            </div>
            {renderRecordList(history, 'No translation history yet.')}
          </div>
        );
      case 'saved':
        return (
          <div className="mb-6 rounded-lg border border-gray-200 bg-white p-6">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg">Saved Results</h2>
              <button
                onClick={() => setSaved([])}
                className="rounded border border-gray-300 px-3 py-1.5 text-xs hover:bg-gray-50"
              >
                Clear Saved
              </button>
            </div>
            {renderRecordList(saved, 'No saved results yet.')}
          </div>
        );
      case 'analytics':
        return (
          <div className="mb-6 rounded-lg border border-gray-200 bg-white p-6">
            <h2 className="mb-3 text-lg">Analytics</h2>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
              <div className="rounded-lg border border-gray-200 p-3">
                <div className="text-xs text-gray-500">Total Predictions</div>
                <div className="text-xl">{analytics.total}</div>
              </div>
              <div className="rounded-lg border border-gray-200 p-3">
                <div className="text-xs text-gray-500">Saved Results</div>
                <div className="text-xl">{analytics.saved}</div>
              </div>
              <div className="rounded-lg border border-gray-200 p-3">
                <div className="text-xs text-gray-500">Average Confidence</div>
                <div className="text-xl">{(analytics.averageConfidence * 100).toFixed(1)}%</div>
              </div>
            </div>
            <div className="mt-4 rounded-lg border border-gray-200 p-3">
              <div className="mb-2 text-sm text-gray-600">Top Predicted Labels</div>
              {analytics.topLabels.length === 0 ? (
                <div className="text-sm text-gray-500">Run a few translations to see trends.</div>
              ) : (
                <div className="space-y-1">
                  {analytics.topLabels.map(([label, count]) => (
                    <div key={label} className="text-sm">
                      {label}: {count}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      case 'documentation':
        return (
          <div className="mb-6 rounded-lg border border-gray-200 bg-white p-6 text-sm text-gray-700">
            <h2 className="mb-3 text-lg text-black">Documentation</h2>
            <div>1. Start backend (`uvicorn`) and frontend (`npm run dev`).</div>
            <div>2. Upload a sign video or record with camera.</div>
            <div>3. Click Start Translation to send it to `/predict`.</div>
            <div>4. Save useful outputs or export CSV for analysis.</div>
            <div className="mt-3 rounded bg-gray-50 p-3 font-mono text-xs text-gray-600">
              MODEL_DATASET=Dataset MODEL_TYPE=transformer MODEL_TRANSFORMER_SIZE=large
            </div>
          </div>
        );
      case 'help':
        return (
          <div className="mb-6 rounded-lg border border-gray-200 bg-white p-6 text-sm text-gray-700">
            <h2 className="mb-3 text-lg text-black">Help Center</h2>
            <div>1. If translation fails, confirm backend is reachable on the configured port.</div>
            <div>2. If camera fails, allow browser camera permission and retry.</div>
            <div>3. If no output appears, check backend logs for model or media errors.</div>
            <div>4. Use Export Results to share reproducible prediction history.</div>
          </div>
        );
      case 'settings':
        return (
          <div className="mb-6 rounded-lg border border-gray-200 bg-white p-6">
            <h2 className="mb-4 text-lg">Settings</h2>
            <div className="mb-4 text-sm text-gray-700">Language: English</div>
            <div className="flex gap-2">
              <button
                onClick={() => setHistory([])}
                className="rounded border border-gray-300 px-3 py-2 text-xs hover:bg-gray-50"
              >
                Clear History
              </button>
              <button
                onClick={() => setSaved([])}
                className="rounded border border-gray-300 px-3 py-2 text-xs hover:bg-gray-50"
              >
                Clear Saved
              </button>
            </div>
          </div>
        );
      case 'new':
      default:
        return null;
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar
        activeSection={activeSection}
        historyCount={history.length}
        savedCount={saved.length}
        onReset={handleReset}
        onSelectSection={setActiveSection}
      />

      <main className="flex-1 overflow-auto">
        <div className="mx-auto max-w-7xl p-8">
          {/* Header */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="mb-2 text-3xl">Indian Sign Language Translator</h1>
                <p className="text-gray-600">Real-time ISL to English translation</p>
              </div>
              <div className="flex gap-3">
                <div className="rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm text-gray-700">
                  Language: English
                </div>
                <button
                  onClick={handleExportResults}
                  className="rounded-lg bg-black px-4 py-2 text-white hover:bg-gray-800"
                >
                  Export Results
                </button>
              </div>
            </div>
          </div>

          {notice && (
            <div className="mb-6 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
              {notice}
            </div>
          )}

          {renderActiveSection()}

          {/* Video Input Section */}
          <VideoInput
            onVideoUpload={handleVideoUpload}
            videoFile={videoFile}
            isProcessing={isProcessing}
            onToggleCamera={handleToggleCamera}
            cameraStream={cameraStream}
            previewUrl={previewUrl}
          />

          {/* Processing Pipeline */}
          <ProcessingPipeline currentStep={currentStep} isProcessing={isProcessing} />

          {/* Keypoints Display */}
          <KeypointsDisplay />

          {/* Translation Output */}
          <TranslationOutput
            currentStep={currentStep}
            isProcessing={isProcessing}
            onStartTranslation={handleStartTranslation}
            onReset={handleReset}
            onCopy={handleCopyPrediction}
            onSaveResult={handleSaveCurrentResult}
            canStart={Boolean(videoFile || cameraStream)}
            prediction={prediction}
            hasSavedCurrent={hasSavedCurrent}
            error={error}
          />
        </div>
      </main>
    </div>
  );
}
