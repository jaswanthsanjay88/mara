"use client";

import React, { useState, useEffect, useRef } from "react";
import { CircleAlert } from "lucide-react";

export default function ToolEditor({ initialTool, existingNames, onSave, onCancel }) {
  const isNew = !initialTool;
  const [name, setName] = useState(initialTool?.name || "");
  const [description, setDescription] = useState(initialTool?.description || "");
  const [paramText, setParamText] = useState(
    initialTool?.parameters ? JSON.stringify(initialTool.parameters, null, 2) : "{\n\n}"
  );

  const [nameError, setNameError] = useState(null);
  const [paramError, setParamError] = useState(null);

  const nameInputRef = useRef(null);

  useEffect(() => {
    nameInputRef.current?.focus();
  }, []);

  const validateName = (val) => {
    const trimmed = val.trim();
    if (!trimmed) {
      setNameError("Name cannot be empty.");
      return false;
    }
    const otherNames = isNew
      ? existingNames
      : existingNames.filter((n) => n !== initialTool.name);
    if (otherNames.includes(trimmed)) {
      setNameError(`A tool named "${trimmed}" already exists. Pick a different name.`);
      return false;
    }
    setNameError(null);
    return true;
  };

  const validateParams = (val) => {
    try {
      const parsed = JSON.parse(val);
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        setParamError("Parameters must be a JSON object.");
        return false;
      }
      setParamError(null);
      return true;
    } catch (err) {
      // Extract position or line number if available
      const msg = err.message.replace(/^JSON\.parse:\s*/, "");
      setParamError(`Parameters must be valid JSON. ${msg}.`);
      return false;
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      onCancel();
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const isNameValid = validateName(name);
    const isParamValid = validateParams(paramText);
    if (!isNameValid || !isParamValid) return;

    try {
      const parsedParams = JSON.parse(paramText);
      onSave({
        name: name.trim(),
        description: description.trim(),
        parameters: parsedParams,
      });
    } catch {
      // Ignored since validateParams caught it
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      onKeyDown={handleKeyDown}
      className="surface border border-[var(--line-strong)] rounded-[10px] p-4 bg-[var(--fill-1)] space-y-3 text-[14px] leading-[22px] transition-all"
    >
      {/* Field: Name */}
      <div className="space-y-1">
        <label className="block text-[13px] leading-[18px] text-[var(--fg)]">
          Name
        </label>
        <input
          ref={nameInputRef}
          type="text"
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            if (nameError) validateName(e.target.value);
          }}
          onBlur={() => validateName(name)}
          placeholder="tool_name"
          className="w-full h-9 px-3 rounded-[6px] border border-[var(--line-strong)] bg-[var(--bg)] text-[var(--fg)] font-mono text-[13px] leading-[20px] focus:outline-none focus:border-[var(--fg)]"
        />
        {nameError && (
          <div className="flex items-center space-x-1.5 text-[13px] leading-[18px] text-[var(--fg)] pt-1">
            <CircleAlert size={14} strokeWidth={1.5} className="shrink-0" />
            <span>{nameError}</span>
          </div>
        )}
      </div>

      {/* Field: Description */}
      <div className="space-y-1">
        <label className="block text-[13px] leading-[18px] text-[var(--fg)]">
          Description
        </label>
        <textarea
          rows={3}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="What the tool does..."
          className="w-full p-2.5 rounded-[6px] border border-[var(--line-strong)] bg-[var(--bg)] text-[var(--fg)] text-[14px] leading-[22px] focus:outline-none focus:border-[var(--fg)]"
        />
        <div className="text-[12px] leading-[16px] text-[var(--fg-2)]">
          Describe what the tool does in one plain sentence. The model reads this to decide when to call it.
        </div>
      </div>

      {/* Field: Parameters JSON */}
      <div className="space-y-1">
        <label className="block text-[13px] leading-[18px] text-[var(--fg)]">
          Parameters
        </label>
        <textarea
          rows={6}
          value={paramText}
          onChange={(e) => {
            setParamText(e.target.value);
            if (paramError) validateParams(e.target.value);
          }}
          onBlur={() => validateParams(paramText)}
          className="w-full p-2.5 rounded-[6px] border border-[var(--line-strong)] bg-[var(--bg)] text-[var(--fg)] font-mono text-[13px] leading-[20px] focus:outline-none focus:border-[var(--fg)]"
        />
        {paramError && (
          <div className="flex items-center space-x-1.5 text-[13px] leading-[18px] text-[var(--fg)] pt-1">
            <CircleAlert size={14} strokeWidth={1.5} className="shrink-0" />
            <span>{paramError}</span>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center justify-end space-x-2 pt-1">
        <button
          type="button"
          onClick={onCancel}
          className="h-8 px-3 rounded-[6px] border border-[var(--line-strong)] text-[13px] leading-[18px] text-[var(--fg)] hover:bg-[var(--fill-2)] transition-colors cursor-pointer"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={!name.trim() || !!nameError || !!paramError}
          className="h-8 px-3 rounded-[6px] bg-[var(--fg)] text-[var(--bg)] text-[13px] leading-[18px] font-[500] hover:opacity-88 active:opacity-76 disabled:bg-[var(--fill-2)] disabled:text-[var(--fg-3)] disabled:cursor-not-allowed transition-opacity cursor-pointer"
        >
          Save tool
        </button>
      </div>
    </form>
  );
}
