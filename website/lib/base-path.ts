const DEPLOY_BASE_PATH = '/flopscope';

export function withBasePath(href: string): string {
  if (!href) return process.env.NODE_ENV === 'production' ? DEPLOY_BASE_PATH : '/';
  if (/^(https?:|mailto:|tel:|#)/.test(href)) return href;

  const normalized = href.startsWith('/') ? href : `/${href}`;
  if (process.env.NODE_ENV !== 'production') {
    return normalized;
  }

  if (normalized === DEPLOY_BASE_PATH || normalized.startsWith(`${DEPLOY_BASE_PATH}/`)) {
    return normalized;
  }

  return `${DEPLOY_BASE_PATH}${normalized}`;
}
