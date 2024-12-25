import { open } from 'node:fs/promises';
import { getData, getFilteredData } from './extractor.mjs'
import * as XLSX from 'xlsx';
import * as fs from 'fs';

function filter1(value, key, map) {
    this.push({ calling_nbr: key, avg_call_count: (value.length / 29).toFixed(3) });
}

function filter2(value, key, map) {
    let seg = {};
    for (let i = 1; i <= 8; i++) {
        seg[`时间段${i}`] = 0;
    }
    value.forEach(ele => {
        let h = parseInt(ele.start_time.split(':')[0]);
        seg[`时间段${Math.floor(h / 3) + 1}`]++;
    });
    let sum = 0;
    Object.values(seg).forEach((val) => { sum += val });
    Object.keys(seg).forEach((val) => { seg[val] /= sum; seg[val] = `${(seg[val] * 100).toFixed(2)}%`; });
    this.push({ calling_nbr: key, ...seg });
}

let [data, cnt] = await getData('./data/data.txt');
console.log(data.size, cnt);

XLSX.set_fs(fs);

let data_filterBy_count = [];
let data_filterBy_time = [];
data.forEach(filter2, data_filterBy_time);
data.forEach(filter1, data_filterBy_count);
const worksheet1 = XLSX.utils.json_to_sheet(data_filterBy_count);
const worksheet2 = XLSX.utils.json_to_sheet(data_filterBy_time);
const workbook = XLSX.utils.book_new();
XLSX.utils.book_append_sheet(workbook, worksheet1, "Counts");
XLSX.utils.book_append_sheet(workbook, worksheet2, "Time");
XLSX.writeFile(workbook, "ans.xlsx");

let filtered_data = await getFilteredData('./data/data.txt');
const columns = Object.keys(filtered_data[0]);
const header = columns.join('\t');
const rows = filtered_data.map(row => {
    return columns.map(column => row[column]).join('\t');
});
const output = [header, ...rows].join('\n');
fs.writeFileSync('output.txt', output, 'utf-8');