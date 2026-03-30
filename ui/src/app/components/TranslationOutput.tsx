import { Copy, Info, MessageSquareText, Play, RotateCcw, Save } from 'lucide-react';

interface TranslationOutputProps {
  currentStep: number;
  isProcessing: boolean;
  onStartTranslation: () => void;
  onReset: () => void;
  onCopy: () => void;
  onSaveResult: () => void;
  canStart: boolean;
  prediction: {
    label: string;
    score: number;
    elapsed_ms: number;
  } | null;
  hasSavedCurrent: boolean;
  error: string | null;
}

export function TranslationOutput({
  currentStep,
  isProcessing,
  onStartTranslation,
  onReset,
  onCopy,
  onSaveResult,
  canStart,
  prediction,
  hasSavedCurrent,
  error,
}: TranslationOutputProps) {
  const isComplete = Boolean(prediction) && !isProcessing;

  const confidencePercent = prediction ? (prediction.score * 100).toFixed(1) : '--';
  const confidenceLevel = prediction
    ? prediction.score >= 0.85
      ? 'High'
      : prediction.score >= 0.6
        ? 'Medium'
        : 'Low'
    : '--';
  const processingTime = prediction
    ? `${(prediction.elapsed_ms / 1000).toFixed(2)}s`
    : '--s';

  return (
    <div className="mb-6 rounded-lg border border-gray-200 bg-white p-6">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="mb-1 text-xl">Translation Output</h2>
          <p className="text-sm text-gray-600">Translated English text with confidence score</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={onCopy}
            disabled={!prediction}
            className={`flex items-center gap-1.5 rounded border px-3 py-1.5 text-xs ${
              prediction
                ? 'border-gray-300 hover:bg-gray-50'
                : 'cursor-not-allowed border-gray-200 bg-gray-100 text-gray-400'
            }`}
          >
            <Copy className="h-3.5 w-3.5" />
            Copy
          </button>
        </div>
      </div>

      <div className="mb-6 flex min-h-[160px] flex-col items-center justify-center rounded-lg border border-gray-200 bg-gray-50 p-10">
        {error ? (
          <div className="text-center">
            <div className="mb-2 text-2xl text-red-600">Error</div>
            <div className="text-sm text-gray-500">{error}</div>
          </div>
        ) : isComplete && prediction ? (
          <div className="text-center">
            <div className="mb-2 text-3xl">"{prediction.label}"</div>
            <div className="text-sm text-gray-500">Translation complete</div>
          </div>
        ) : currentStep >= 3 ? (
          <div className="text-center">
            <MessageSquareText className="mx-auto mb-2 h-8 w-8 text-gray-300" />
            <div className="text-gray-500">Translation in progress...</div>
            <div className="text-xs text-gray-400">Processing through Transformer model</div>
          </div>
        ) : (
          <div className="text-center">
            <MessageSquareText className="mx-auto mb-2 h-8 w-8 text-gray-300" />
            <div className="text-gray-400">Translation will appear here</div>
            <div className="text-xs text-gray-400">Start processing to see results</div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="rounded-lg border border-gray-200 p-4">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-sm text-gray-600">Accuracy Score</div>
              <Info
                className="h-3.5 w-3.5 text-gray-400"
                title="Model confidence for the top predicted label"
              />
            </div>
            <div className="text-2xl font-medium">{prediction ? `${confidencePercent}%` : '--%'}</div>
          </div>

        <div className="rounded-lg border border-gray-200 p-4">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-sm text-gray-600">Confidence Level</div>
              <Info
                className="h-3.5 w-3.5 text-gray-400"
                title="High: >=85%, Medium: >=60%, Low: <60%"
              />
            </div>
            <div className="text-2xl font-medium">{confidenceLevel}</div>
          </div>

        <div className="rounded-lg border border-gray-200 p-4">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-sm text-gray-600">Processing Time</div>
              <Info
                className="h-3.5 w-3.5 text-gray-400"
                title="End-to-end backend response time for this prediction"
              />
            </div>
            <div className="text-2xl font-medium">{processingTime}</div>
          </div>
      </div>

      <div className="mt-6 flex items-center justify-between border-t border-gray-200 pt-6">
        <button
          onClick={onReset}
          className="flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-6 py-2.5 hover:bg-gray-50"
        >
          <RotateCcw className="h-4 w-4" />
          Reset
        </button>

        <div className="flex gap-3">
          <button
            onClick={onSaveResult}
            disabled={!isComplete}
            className={`flex items-center gap-2 rounded-lg border px-6 py-2.5 ${
              isComplete
                ? 'border-gray-300 bg-white hover:bg-gray-50'
                : 'cursor-not-allowed border-gray-200 bg-gray-100 text-gray-400'
            }`}
          >
            <Save className="h-4 w-4" />
            {hasSavedCurrent ? 'Saved' : 'Save Results'}
          </button>

          <button
            onClick={onStartTranslation}
            disabled={!canStart || isProcessing}
            className={`flex items-center gap-2 rounded-lg px-6 py-2.5 ${
              !canStart || isProcessing
                ? 'cursor-not-allowed bg-gray-300 text-gray-500'
                : 'bg-black text-white hover:bg-gray-800'
            }`}
          >
            <Play className="h-4 w-4" />
            {isProcessing ? 'Processing...' : 'Start Translation'}
          </button>
        </div>
      </div>
    </div>
  );
}
