// =========================================================================
// Flopscope Design System — Shiki themes
// Derived from the system palette: coral = value, ink-gray = structure.
//   string     → coral-hover  (#D23934)   — AA at body size
//   operator   → coral        (#F0524D)   — brand accent
//   function   → V-blue       (#2959C4)   — info, deepened
//   class/type → warning      (#FA9E33)   — reused from op-tag
//   keyword    → violet       (#5E36C4)
//   number     → teal         (#0B6D7A)
//   comment    → gray-400 italic
//   ink        → gray-900
//   punctuation→ gray-600
// Canvas: pure white light / neutral #1A1D1E dark (NOT warm-brown).
// =========================================================================

type FlopscopeTheme = {
  name: string;
  type: 'light' | 'dark';
  colors: Record<string, string>;
  tokenColors: ReadonlyArray<{
    scope: string | ReadonlyArray<string>;
    settings: { foreground?: string; background?: string; fontStyle?: string };
  }>;
};

export const flopscopeLight: FlopscopeTheme = {
  name: 'flopscope-paper',
  type: 'light',
  colors: {
    'editor.background': '#FFFFFF',
    'editor.foreground': '#292C2D',
    'editorLineNumber.foreground': '#AAACAD',
    'editor.selectionBackground': '#FEF2F1',
  },
  tokenColors: [
    { scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: '#AAACAD', fontStyle: 'italic' } },
    { scope: ['string', 'string.quoted', 'string.template'], settings: { foreground: '#D23934' } },
    { scope: ['constant.character.escape', 'constant.other.placeholder'], settings: { foreground: '#F0524D' } },
    { scope: ['constant.numeric', 'constant.language'], settings: { foreground: '#0B6D7A' } },
    { scope: ['keyword', 'storage.type', 'storage.modifier', 'keyword.control'], settings: { foreground: '#5E36C4' } },
    { scope: ['keyword.operator', 'punctuation.separator', 'punctuation.accessor'], settings: { foreground: '#F0524D' } },
    { scope: ['entity.name.function', 'support.function', 'meta.function-call'], settings: { foreground: '#2959C4' } },
    { scope: ['entity.name.class', 'entity.name.type', 'support.class', 'support.type'], settings: { foreground: '#FA9E33' } },
    { scope: ['entity.name.tag'], settings: { foreground: '#2959C4' } },
    { scope: ['entity.other.attribute-name'], settings: { foreground: '#FA9E33' } },
    { scope: ['variable', 'variable.other', 'meta.definition.variable'], settings: { foreground: '#292C2D' } },
    { scope: ['variable.parameter'], settings: { foreground: '#292C2D' } },
    { scope: ['punctuation', 'meta.brace', 'meta.delimiter'], settings: { foreground: '#5D5F60' } },
    { scope: ['markup.heading'], settings: { foreground: '#292C2D', fontStyle: 'bold' } },
    { scope: ['markup.italic'], settings: { fontStyle: 'italic' } },
    { scope: ['markup.bold'], settings: { fontStyle: 'bold' } },
    { scope: ['markup.inserted'], settings: { foreground: '#23B761' } },
    { scope: ['markup.deleted'], settings: { foreground: '#D23934' } },
    { scope: ['invalid'], settings: { foreground: '#D23934' } },
  ],
};

export const flopscopeDark: FlopscopeTheme = {
  name: 'flopscope-ink',
  type: 'dark',
  colors: {
    'editor.background': '#1A1D1E',
    'editor.foreground': '#E8EAEB',
    'editorLineNumber.foreground': '#6B7072',
    'editor.selectionBackground': '#2B2F30',
  },
  tokenColors: [
    { scope: ['comment', 'punctuation.definition.comment'], settings: { foreground: '#6B7072', fontStyle: 'italic' } },
    { scope: ['string', 'string.quoted', 'string.template'], settings: { foreground: '#FF8A7A' } },
    { scope: ['constant.character.escape', 'constant.other.placeholder'], settings: { foreground: '#F0524D' } },
    { scope: ['constant.numeric', 'constant.language'], settings: { foreground: '#6BD4CE' } },
    { scope: ['keyword', 'storage.type', 'storage.modifier', 'keyword.control'], settings: { foreground: '#B99BFF' } },
    { scope: ['keyword.operator', 'punctuation.separator', 'punctuation.accessor'], settings: { foreground: '#F0524D' } },
    { scope: ['entity.name.function', 'support.function', 'meta.function-call'], settings: { foreground: '#8FB4FF' } },
    { scope: ['entity.name.class', 'entity.name.type', 'support.class', 'support.type'], settings: { foreground: '#FFC27A' } },
    { scope: ['entity.name.tag'], settings: { foreground: '#8FB4FF' } },
    { scope: ['entity.other.attribute-name'], settings: { foreground: '#FFC27A' } },
    { scope: ['variable', 'variable.other', 'meta.definition.variable'], settings: { foreground: '#E8EAEB' } },
    { scope: ['variable.parameter'], settings: { foreground: '#E8EAEB' } },
    { scope: ['punctuation', 'meta.brace', 'meta.delimiter'], settings: { foreground: '#A8ACAD' } },
    { scope: ['markup.heading'], settings: { foreground: '#E8EAEB', fontStyle: 'bold' } },
    { scope: ['markup.italic'], settings: { fontStyle: 'italic' } },
    { scope: ['markup.bold'], settings: { fontStyle: 'bold' } },
    { scope: ['markup.inserted'], settings: { foreground: '#23B761' } },
    { scope: ['markup.deleted'], settings: { foreground: '#FF8A7A' } },
    { scope: ['invalid'], settings: { foreground: '#FF8A7A' } },
  ],
};
