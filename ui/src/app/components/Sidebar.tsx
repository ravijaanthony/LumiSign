export type SidebarSection =
  | 'new'
  | 'history'
  | 'saved'
  | 'analytics'
  | 'documentation'
  | 'help'
  | 'settings';

interface SidebarProps {
  activeSection: SidebarSection;
  historyCount: number;
  savedCount: number;
  onReset: () => void;
  onSelectSection: (section: SidebarSection) => void;
}

export function Sidebar({
  activeSection,
  historyCount,
  savedCount,
  onReset,
  onSelectSection,
}: SidebarProps) {
  const navButtonClass = (section: SidebarSection) =>
    `w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left ${
      activeSection === section
        ? 'bg-gray-100 text-gray-900'
        : 'text-gray-600 hover:bg-gray-100'
    }`;

  return (
    <aside className="w-64 bg-white border-r border-gray-200 p-6 flex flex-col">
      {/* Logo */}
      <div className="flex items-center gap-3 mb-8">
        <div className="w-10 h-10 bg-black rounded-lg flex items-center justify-center">
          <span className="text-white text-xl">🤟</span>
        </div>
        <span className="font-semibold">LumiSign</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1">
        <div className="space-y-2">
          <button
            onClick={() => {
              onReset();
              onSelectSection('new');
            }}
            className={navButtonClass('new')}
          >
            <span>📹</span>
            <span>New Translation</span>
          </button>

          <button
            onClick={() => onSelectSection('history')}
            className={navButtonClass('history')}
          >
            <span>🕐</span>
            <span className="flex-1">History</span>
            <span className="rounded bg-gray-200 px-2 py-0.5 text-xs text-gray-700">
              {historyCount}
            </span>
          </button>

          <button
            onClick={() => onSelectSection('saved')}
            className={navButtonClass('saved')}
          >
            <span>🔖</span>
            <span className="flex-1">Saved</span>
            <span className="rounded bg-gray-200 px-2 py-0.5 text-xs text-gray-700">
              {savedCount}
            </span>
          </button>

          <button
            onClick={() => onSelectSection('analytics')}
            className={navButtonClass('analytics')}
          >
            <span>📊</span>
            <span>Analytics</span>
          </button>
        </div>

        {/* Resources Section */}
        <div className="mt-8">
          <div className="text-xs uppercase text-gray-400 mb-3 px-4">Resources</div>
          <div className="space-y-2">
            <button
              onClick={() => onSelectSection('documentation')}
              className={navButtonClass('documentation')}
            >
              <span>📚</span>
              <span>Documentation</span>
            </button>

            <button
              onClick={() => onSelectSection('help')}
              className={navButtonClass('help')}
            >
              <span>❓</span>
              <span>Help Center</span>
            </button>

            <button
              onClick={() => onSelectSection('settings')}
              className={navButtonClass('settings')}
            >
              <span>⚙️</span>
              <span>Settings</span>
            </button>
          </div>
        </div>
      </nav>

      {/* User Profile */}
      <div className="pt-6 border-t border-gray-200">
        <div className="flex items-center gap-3 px-4 py-3">
          <div className="w-10 h-10 bg-gray-300 rounded-full flex items-center justify-center">
            <span>👤</span>
          </div>
          <div className="flex-1">
            <div className="font-medium">Local Session</div>
            <div className="text-sm text-gray-500">Stored in this browser</div>
          </div>
          <button
            onClick={() => onSelectSection('settings')}
            className="text-gray-400 hover:text-gray-600"
            aria-label="Open settings"
          >
            ⋮
          </button>
        </div>
      </div>
    </aside>
  );
}
