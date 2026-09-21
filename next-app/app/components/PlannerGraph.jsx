"use client";

import React from "react";
import { Check } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";

export default function PlannerGraph({ planData, totalLatency }) {
  const reduceMotion = useReducedMotion();

  if (!planData || !planData.steps || planData.steps.length === 0) {
    return null;
  }

  const { steps, goal, total_steps } = planData;

  return (
    <div className="bg-[#0a0b10] border border-[#181a24] rounded-md p-4 space-y-3">
      <div className="flex items-center justify-between text-xs font-mono text-neutral-400">
        <div className="flex items-center space-x-2">
          <span className="text-neutral-200 font-medium">{goal || "Execution Plan"}</span>
          <span className="text-neutral-600">·</span>
          <span>{total_steps || steps.length} steps</span>
        </div>
        <div>
          <span>{totalLatency || 0}ms total</span>
        </div>
      </div>

      <div className="divide-y divide-[#141620] border-t border-[#141620]">
        {steps.map((stepItem, idx) => {
          const tool = stepItem.tool || "none";
          const latency = stepItem.inference_latency_ms || 0;
          const conf = Math.round((stepItem.confidence || 1.0) * 100);

          return (
            <motion.div
              key={idx}
              initial={reduceMotion ? false : { opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.15, delay: reduceMotion ? 0 : idx * 0.03 }}
              className="py-2.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs font-mono"
            >
              <div className="flex items-center space-x-3">
                <span className="text-neutral-500 w-4">{idx + 1}.</span>
                <span className="text-neutral-200">"{stepItem.sub_query}"</span>
              </div>

              <div className="flex items-center space-x-3 self-end sm:self-center">
                <span className="text-amber-400/90 px-1.5 py-0.5 rounded bg-amber-500/10 text-[11px]">
                  {tool}
                </span>
                <span className="text-neutral-500 text-[11px]">{conf}%</span>
                <span className="text-neutral-600">·</span>
                <span className="text-neutral-400 text-[11px]">{latency}ms</span>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
