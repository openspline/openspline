import {copyFile,mkdir,readdir,rm} from 'node:fs/promises';
const dest='packages/server/src/openspline_server/static/browser';
await mkdir(dest,{recursive:true});
for(const name of await readdir(dest))if(name.endsWith('.d.ts'))await rm(`${dest}/${name}`);
for(const name of await readdir('packages/browser/dist'))if(name.endsWith('.js'))await copyFile(`packages/browser/dist/${name}`,`${dest}/${name}`);
