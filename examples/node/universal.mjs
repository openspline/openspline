import {Openspline} from '@openspline/node';
const avatar=await new Openspline().avatar(process.argv[2]);
try {console.log(avatar.viewerUrl);await avatar.playFile(process.argv[3]);}
finally {await avatar.close();}
