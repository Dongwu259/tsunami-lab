import { TsunamiApp } from './app';

try {
  new TsunamiApp(document.getElementById('app') as HTMLElement);
} catch (err) {
  const fatal = document.getElementById('fatal');
  if (fatal) {
    fatal.style.display = 'grid';
    fatal.innerHTML =
      '初始化失败:当前浏览器可能不支持所需的 WebGL 浮点纹理能力。<br />' +
      '请使用最新版 Chrome / Edge / Safari / Firefox 重试。<br />' +
      `<span style="font-size:12px;opacity:.6">${String(err)}</span>`;
  }
  console.error(err);
}
