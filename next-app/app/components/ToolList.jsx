"use client";

import React, { useState } from "react";
import { Plus } from "lucide-react";
import ToolCard from "./ToolCard";
import ToolEditor from "./ToolEditor";

export default function ToolList({
  tools,
  justCalledToolNames = [],
  onToggleTool,
  onUpdateTool,
  onRemoveTool,
  onAddTool,
}) {
  const [editingToolName, setEditingToolName] = useState(null);
  const [isAddingNew, setIsAddingNew] = useState(false);

  const existingNames = tools.map((t) => t.name);

  return (
    <div className="flex flex-col h-full space-y-2.5">
      {/* Tools Stack - compact and scrollable if needed */}
      <div className="flex-1 space-y-2 overflow-y-auto no-scrollbar pr-0.5">
        {tools.length === 0 && !isAddingNew ? (
          <div className="py-6 text-[13px] leading-[20px] text-[var(--fg-2)]">
            No tools yet. The model can only call tools you declare.
          </div>
        ) : (
          tools.map((tool) => {
            if (editingToolName === tool.name) {
              return (
                <ToolEditor
                  key={tool.name}
                  initialTool={tool}
                  existingNames={existingNames}
                  onSave={(updated) => {
                    onUpdateTool(tool.name, updated);
                    setEditingToolName(null);
                  }}
                  onCancel={() => setEditingToolName(null)}
                />
              );
            }

            const isJustCalled = justCalledToolNames.includes(tool.name);

            return (
              <ToolCard
                key={tool.name}
                tool={tool}
                isJustCalled={isJustCalled}
                onToggle={(toolName, enabled) => onToggleTool?.(toolName, enabled)}
                onEdit={() => {
                  setIsAddingNew(false);
                  setEditingToolName(tool.name);
                }}
                onRemove={() => onRemoveTool(tool.name)}
              />
            );
          })
        )}

        {isAddingNew && (
          <ToolEditor
            existingNames={existingNames}
            onSave={(newTool) => {
              onAddTool(newTool);
              setIsAddingNew(false);
            }}
            onCancel={() => setIsAddingNew(false)}
          />
        )}
      </div>

      {/* Add Tool Ghost Button */}
      {!isAddingNew && (
        <button
          type="button"
          onClick={() => {
            setEditingToolName(null);
            setIsAddingNew(true);
          }}
          className="w-full h-8 rounded-[6px] border border-[var(--line)] hover:border-[var(--line-strong)] flex items-center justify-center space-x-1.5 text-[12px] leading-[18px] font-normal text-[var(--fg)] hover:bg-[var(--fill-2)] transition-colors cursor-pointer"
        >
          <Plus size={13} strokeWidth={1.5} />
          <span>Add tool</span>
        </button>
      )}
    </div>
  );
}
