import { Progress } from './ui/progress';

interface ProcessingPipelineProps {
  currentStep: number;
  isProcessing: boolean;
}

export function ProcessingPipeline({ currentStep, isProcessing }: ProcessingPipelineProps) {
  const steps = [
    {
      id: 1,
      title: 'Step 1: Video Enhancement',
      description: 'Brightening darkened video data',
    },
    {
      id: 2,
      title: 'Step 2: Keypoint Extraction',
      description: 'Detecting hand and body landmarks',
    },
    {
      id: 3,
      title: 'Step 3: Transformer Translation',
      description: 'Sequence classification using the trained transformer model',
    },
    {
      id: 4,
      title: 'Overall Progress',
      description: 'Total processing completion',
    },
  ];

  const getStepStatus = (stepId: number) => {
    if (stepId === 4) {
      if (currentStep >= 4) return 'completed';
      if (currentStep > 0) return 'processing';
      return 'pending';
    }
    if (currentStep > stepId) return 'completed';
    if (currentStep === stepId) return 'processing';
    return 'pending';
  };

  const getStatusIcon = (stepId: number) => {
    const status = getStepStatus(stepId);
    if (status === 'completed') return '✓';
    if (status === 'processing') return '⟳';
    return '○';
  };

  const getStatusColor = (stepId: number) => {
    const status = getStepStatus(stepId);
    if (status === 'completed') return 'text-green-600 bg-green-50';
    if (status === 'processing') return 'text-blue-600 bg-blue-50';
    return 'text-gray-400 bg-gray-100';
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl mb-1">Processing Pipeline</h2>
          <p className="text-sm text-gray-600">Real-time translation progress and status</p>
        </div>
        <div className={`px-3 py-1.5 rounded-lg text-sm ${
          currentStep === 0 ? 'bg-gray-100 text-gray-600' : 
          isProcessing ? 'bg-blue-50 text-blue-600' : 
          'bg-green-50 text-green-600'
        }`}>
          {currentStep === 0 ? 'Idle' : isProcessing ? 'Processing...' : 'Completed'}
        </div>
      </div>

      <div className="space-y-4">
        {steps.map((step, index) => (
          <div 
            key={step.id}
            className={`border rounded-lg p-4 transition-all ${
              getStepStatus(step.id) === 'processing' ? 'border-blue-300 bg-blue-50' : 
              getStepStatus(step.id) === 'completed' ? 'border-green-300 bg-green-50' : 
              'border-gray-200 bg-white'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-3 flex-1">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-lg ${getStatusColor(step.id)}`}>
                  {getStatusIcon(step.id)}
                </div>
                <div className="flex-1">
                  <div className="font-medium mb-1">{step.title}</div>
                  <div className="text-sm text-gray-600">{step.description}</div>
                  
                  {getStepStatus(step.id) !== 'pending' && step.id !== 4 && (
                    <div className="mt-3">
                      <Progress 
                        value={getStepStatus(step.id) === 'completed' ? 100 : 60} 
                        className="h-2"
                      />
                    </div>
                  )}

                  {step.id === 4 && currentStep > 0 && (
                    <div className="mt-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-600">Progress</span>
                        <span className="text-sm font-medium">
                          {getStepStatus(step.id) === 'completed' ? 100 : Math.min(currentStep * 25, 100)}%
                        </span>
                      </div>
                      <Progress 
                        value={
                          getStepStatus(step.id) === 'completed'
                            ? 100
                            : Math.min(currentStep * 25, 100)
                        } 
                        className="h-2"
                      />
                    </div>
                  )}
                </div>
              </div>
              <div className="text-sm text-gray-500">
                {getStepStatus(step.id) === 'completed' ? 'Done' : 
                 getStepStatus(step.id) === 'processing' ? 'In Progress' : 
                 'Pending'}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
