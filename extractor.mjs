import { open } from 'node:fs/promises';
/**
     * 
     * @param {*} value 
     * @param {*} key 
     * @param {*} map 
     * filter3中需要清洗出的数据包括：user对应的外呼次数，接听次数，8个时间段中的外呼次数，外呼平均通话时长
     */
function filter3(value, key, map) {
    let seg = {};
    for (let i = 1; i <= 8; i++) {
        seg[`TimeSlot${i}`] = 0;
    }
    let incall = 0, outcall = 0, dur = 0;
    value.forEach(ele => {
        let h = parseInt(ele.start_time.split(':')[0]);
        if (ele.active === true) {
            outcall++; //总外呼次数+1
            seg[`TimeSlot${Math.floor(h / 3) + 1}`]++; // 对应时间段外呼次数+1
            dur += parseInt(ele.raw_dur);
        }
        else if (ele.active === false) {
            incall++;//总接听次数+1
            // 不统计时间段中的接听次数
        }
    });
    if (outcall !== 0)
        this.push({ user_id: key, OutgoingCalls: outcall, IncomingCalls: incall, ...seg, AvgDuration: (dur / outcall) });
}
function parse(raw) {
    const caller_key = 1, callee_key = 2;
    let caller = raw[caller_key];
    let callee = raw[callee_key];
    let info = {
        day_id: raw[0],
        calling_optr: raw[3],
        called_optr: raw[4],
        calling_city: raw[5],
        called_city: raw[6],
        calling_roam_city: raw[7],
        called_roam_city: raw[8],
        start_time: raw[9],
        end_time: raw[10],
        raw_dur: raw[11],
        call_type: raw[12],
        calling_cell: raw[13],
        active: true,
    };
    return [caller, callee, info];
}
let s_path = ''
let [data, cnt] = [null, null];
async function extract(path) {
    let data = new Map();
    let cnt = 0;
    const raw_datas = await open(path, 'r', 0o666, (err, fd) => {
        if (err) {
            console.log(err);
            return;
        }
    });
    for await (const raw_data of raw_datas.readLines()) {
        let raw_info = raw_data.split('\t');
        let [caller, callee, info] = parse(raw_info);
        info.active = true;
        if (!data.has(caller)) {
            data.set(caller, [info]);
        } else {
            data.get(caller).push(info);
        }

        let info2 = JSON.parse(JSON.stringify(info));
        info2.active = false;
        if (!data.has(callee)) {
            data.set(callee, [info2]);
        } else {
            data.get(callee).push(info2);
        }

        cnt++;
    }
    return [data, cnt];
}
export async function getData(path) {
    // return [data, cnt];
    if (s_path !== path) {
        [data, cnt] = await extract(path);
        s_path = path;
    }
    return [data, cnt];
}
let filtered_data = []
let filtered = 0;
export async function getFilteredData(path) {
    // let [data, cnt] = await getData('./data/data.txt');
    [data, cnt] = await getData(path);
    if (filtered === 0) {
        data.forEach(filter3, filtered_data);
        filtered = 1;
    }
    return filtered_data;
}
