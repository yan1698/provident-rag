import { useEffect, useId, useRef, useState } from "react";

export default function AgentMenu({
  label,
  value,
  placeholder,
  options,
  onChange,
  className = "",
  invalid = false,
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const root = useRef(null);
  const buttonRef = useRef(null);
  const optionRefs = useRef([]);
  const listId = useId();
  const current = options.find((item) => item.value === value);

  useEffect(() => {
    if (!open) return undefined;
    const index = options.findIndex((item) => item.value === value);
    setActive(index >= 0 ? index : 0);
    const close = (event) => {
      if (root.current && !root.current.contains(event.target)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open, options, value]);

  useEffect(() => {
    if (!open || !options.length) return undefined;
    optionRefs.current[active]?.focus();
    return undefined;
  }, [open, active, options.length]);

  const closeToButton = () => {
    setOpen(false);
    buttonRef.current?.focus();
  };

  const choose = (item) => {
    onChange(item.value);
    closeToButton();
  };

  const onButtonKey = (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      setOpen(true);
    }
    if (event.key === "Escape") {
      event.preventDefault();
      setOpen(false);
    }
  };

  const onOptionKey = (event, index) => {
    if (!options.length) return;
    if (event.key === "Escape") {
      event.preventDefault();
      closeToButton();
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActive((index + 1) % options.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive((index - 1 + options.length) % options.length);
    } else if (event.key === "Home") {
      event.preventDefault();
      setActive(0);
    } else if (event.key === "End") {
      event.preventDefault();
      setActive(options.length - 1);
    } else if (event.key === "Tab") {
      setOpen(false);
    }
  };

  return (
    <div className={`agent-menu${className ? ` ${className}` : ""}`} ref={root}>
      <button
        ref={buttonRef}
        type="button"
        className={`agent-menu-button${invalid ? " invalid" : ""}`}
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        aria-invalid={invalid || undefined}
        onClick={() => setOpen((next) => !next)}
        onKeyDown={onButtonKey}
      >
        {current ? current.label : placeholder}
      </button>
      {open ? (
        <div className="agent-menu-panel" id={listId} role="listbox" aria-label={label}>
          {options.length ? (
            options.map((item, index) => (
              <button
                key={item.value || `empty-${index}`}
                ref={(node) => {
                  optionRefs.current[index] = node;
                }}
                id={`${listId}-opt-${index}`}
                type="button"
                role="option"
                tabIndex={-1}
                aria-selected={item.value === value}
                onClick={() => choose(item)}
                onKeyDown={(event) => onOptionKey(event, index)}
              >
                <span className="agent-menu-label">{item.label}</span>
                {item.hint ? <span className="agent-menu-hint">{item.hint}</span> : null}
              </button>
            ))
          ) : (
            <p className="tiny">没有可选的知识库</p>
          )}
        </div>
      ) : null}
    </div>
  );
}
