/* eslint-disable */

$(document).ready(function() {
var options = {
    controls: true,
    width: 600,
    height: 450,
    fluid: false,
    controlBar: {
        volumePanel: false
    },
    plugins: {
        record: {
            audio: false,
            video: true,
            maxLength: 2,
            debug: true,
            videoFrameRate : 15.0,
            frameWidth: 600,
            frameHeight: 450
        }
    }
};
 let words = ['ABOUT', 'ABSOLUTELY', 'ABUSE', 'ACCESS', 'ACCORDING', 'ACCUSED', 'ACROSS', 'ACTION', 'ACTUALLY', 'AFFAIRS', 'AFFECTED', 'AFRICA', 'AFTER', 'AFTERNOON', 'AGAIN', 'AGAINST', 'AGREE', 'AGREEMENT', 'AHEAD', 'ALLEGATIONS', 'ALLOW', 'ALLOWED', 'ALMOST', 'ALREADY', 'ALWAYS', 'AMERICA', 'AMERICAN', 'AMONG', 'AMOUNT', 'ANNOUNCED', 'ANOTHER', 'ANSWER', 'ANYTHING', 'AREAS', 'AROUND', 'ARRESTED', 'ASKED', 'ASKING', 'ATTACK', 'ATTACKS', 'AUTHORITIES', 'BANKS', 'BECAUSE', 'BECOME', 'BEFORE', 'BEHIND', 'BEING', 'BELIEVE', 'BENEFIT', 'BENEFITS', 'BETTER', 'BETWEEN', 'BIGGEST', 'BILLION', 'BLACK', 'BORDER', 'BRING', 'BRITAIN', 'BRITISH', 'BROUGHT', 'BUDGET', 'BUILD', 'BUILDING', 'BUSINESS', 'BUSINESSES', 'CALLED', 'CAMERON', 'CAMPAIGN', 'CANCER', 'CANNOT', 'CAPITAL', 'CASES', 'CENTRAL', 'CERTAINLY', 'CHALLENGE', 'CHANCE', 'CHANGE', 'CHANGES', 'CHARGE', 'CHARGES', 'CHIEF', 'CHILD', 'CHILDREN', 'CHINA', 'CLAIMS', 'CLEAR', 'CLOSE', 'CLOUD', 'COMES', 'COMING', 'COMMUNITY', 'COMPANIES', 'COMPANY', 'CONCERNS', 'CONFERENCE', 'CONFLICT', 'CONSERVATIVE', 'CONTINUE', 'CONTROL', 'COULD', 'COUNCIL', 'COUNTRIES', 'COUNTRY', 'COUPLE', 'COURSE', 'COURT', 'CRIME', 'CRISIS', 'CURRENT', 'CUSTOMERS', 'DAVID', 'DEATH', 'DEBATE', 'DECIDED', 'DECISION', 'DEFICIT', 'DEGREES', 'DESCRIBED', 'DESPITE', 'DETAILS', 'DIFFERENCE', 'DIFFERENT', 'DIFFICULT', 'DOING', 'DURING', 'EARLY', 'EASTERN', 'ECONOMIC', 'ECONOMY', 'EDITOR', 'EDUCATION', 'ELECTION', 'EMERGENCY', 'ENERGY', 'ENGLAND', 'ENOUGH', 'EUROPE', 'EUROPEAN', 'EVENING', 'EVENTS', 'EVERY', 'EVERYBODY', 'EVERYONE', 'EVERYTHING', 'EVIDENCE', 'EXACTLY', 'EXAMPLE', 'EXPECT', 'EXPECTED', 'EXTRA', 'FACING', 'FAMILIES', 'FAMILY', 'FIGHT', 'FIGHTING', 'FIGURES', 'FINAL', 'FINANCIAL', 'FIRST', 'FOCUS', 'FOLLOWING', 'FOOTBALL', 'FORCE', 'FORCES', 'FOREIGN', 'FORMER', 'FORWARD', 'FOUND', 'FRANCE', 'FRENCH', 'FRIDAY', 'FRONT', 'FURTHER', 'FUTURE', 'GAMES', 'GENERAL', 'GEORGE', 'GERMANY', 'GETTING', 'GIVEN', 'GIVING', 'GLOBAL', 'GOING', 'GOVERNMENT', 'GREAT', 'GREECE', 'GROUND', 'GROUP', 'GROWING', 'GROWTH', 'GUILTY', 'HAPPEN', 'HAPPENED', 'HAPPENING', 'HAVING', 'HEALTH', 'HEARD', 'HEART', 'HEAVY', 'HIGHER', 'HISTORY', 'HOMES', 'HOSPITAL', 'HOURS', 'HOUSE', 'HOUSING', 'HUMAN', 'HUNDREDS', 'IMMIGRATION', 'IMPACT', 'IMPORTANT', 'INCREASE', 'INDEPENDENT', 'INDUSTRY', 'INFLATION', 'INFORMATION', 'INQUIRY', 'INSIDE', 'INTEREST', 'INVESTMENT', 'INVOLVED', 'IRELAND', 'ISLAMIC', 'ISSUE', 'ISSUES', 'ITSELF', 'JAMES', 'JUDGE', 'JUSTICE', 'KILLED', 'KNOWN', 'LABOUR', 'LARGE', 'LATER', 'LATEST', 'LEADER', 'LEADERS', 'LEADERSHIP', 'LEAST', 'LEAVE', 'LEGAL', 'LEVEL', 'LEVELS', 'LIKELY', 'LITTLE', 'LIVES', 'LIVING', 'LOCAL', 'LONDON', 'LONGER', 'LOOKING', 'MAJOR', 'MAJORITY', 'MAKES', 'MAKING', 'MANCHESTER', 'MARKET', 'MASSIVE', 'MATTER', 'MAYBE', 'MEANS', 'MEASURES', 'MEDIA', 'MEDICAL', 'MEETING', 'MEMBER', 'MEMBERS', 'MESSAGE', 'MIDDLE', 'MIGHT', 'MIGRANTS', 'MILITARY', 'MILLION', 'MILLIONS', 'MINISTER', 'MINISTERS', 'MINUTES', 'MISSING', 'MOMENT', 'MONEY', 'MONTH', 'MONTHS', 'MORNING', 'MOVING', 'MURDER', 'NATIONAL', 'NEEDS', 'NEVER', 'NIGHT', 'NORTH', 'NORTHERN', 'NOTHING', 'NUMBER', 'NUMBERS', 'OBAMA', 'OFFICE', 'OFFICERS', 'OFFICIALS', 'OFTEN', 'OPERATION', 'OPPOSITION', 'ORDER', 'OTHER', 'OTHERS', 'OUTSIDE', 'PARENTS', 'PARLIAMENT', 'PARTIES', 'PARTS', 'PARTY', 'PATIENTS', 'PAYING', 'PEOPLE', 'PERHAPS', 'PERIOD', 'PERSON', 'PERSONAL', 'PHONE', 'PLACE', 'PLACES', 'PLANS', 'POINT', 'POLICE', 'POLICY', 'POLITICAL', 'POLITICIANS', 'POLITICS', 'POSITION', 'POSSIBLE', 'POTENTIAL', 'POWER', 'POWERS', 'PRESIDENT', 'PRESS', 'PRESSURE', 'PRETTY', 'PRICE', 'PRICES', 'PRIME', 'PRISON', 'PRIVATE', 'PROBABLY', 'PROBLEM', 'PROBLEMS', 'PROCESS', 'PROTECT', 'PROVIDE', 'PUBLIC', 'QUESTION', 'QUESTIONS', 'QUITE', 'RATES', 'RATHER', 'REALLY', 'REASON', 'RECENT', 'RECORD', 'REFERENDUM', 'REMEMBER', 'REPORT', 'REPORTS', 'RESPONSE', 'RESULT', 'RETURN', 'RIGHT', 'RIGHTS', 'RULES', 'RUNNING', 'RUSSIA', 'RUSSIAN', 'SAYING', 'SCHOOL', 'SCHOOLS', 'SCOTLAND', 'SCOTTISH', 'SECOND', 'SECRETARY', 'SECTOR', 'SECURITY', 'SEEMS', 'SENIOR', 'SENSE', 'SERIES', 'SERIOUS', 'SERVICE', 'SERVICES', 'SEVEN', 'SEVERAL', 'SHORT', 'SHOULD', 'SIDES', 'SIGNIFICANT', 'SIMPLY', 'SINCE', 'SINGLE', 'SITUATION', 'SMALL', 'SOCIAL', 'SOCIETY', 'SOMEONE', 'SOMETHING', 'SOUTH', 'SOUTHERN', 'SPEAKING', 'SPECIAL', 'SPEECH', 'SPEND', 'SPENDING', 'SPENT', 'STAFF', 'STAGE', 'STAND', 'START', 'STARTED', 'STATE', 'STATEMENT', 'STATES', 'STILL', 'STORY', 'STREET', 'STRONG', 'SUNDAY', 'SUNSHINE', 'SUPPORT', 'SYRIA', 'SYRIAN', 'SYSTEM', 'TAKEN', 'TAKING', 'TALKING', 'TALKS', 'TEMPERATURES', 'TERMS', 'THEIR', 'THEMSELVES', 'THERE', 'THESE', 'THING', 'THINGS', 'THINK', 'THIRD', 'THOSE', 'THOUGHT', 'THOUSANDS', 'THREAT', 'THREE', 'THROUGH', 'TIMES', 'TODAY', 'TOGETHER', 'TOMORROW', 'TONIGHT', 'TOWARDS', 'TRADE', 'TRIAL', 'TRUST', 'TRYING', 'UNDER', 'UNDERSTAND', 'UNION', 'UNITED', 'UNTIL', 'USING', 'VICTIMS', 'VIOLENCE', 'VOTERS', 'WAITING', 'WALES', 'WANTED', 'WANTS', 'WARNING', 'WATCHING', 'WATER', 'WEAPONS', 'WEATHER', 'WEEKEND', 'WEEKS', 'WELCOME', 'WELFARE', 'WESTERN', 'WESTMINSTER', 'WHERE', 'WHETHER', 'WHICH', 'WHILE', 'WHOLE', 'WINDS', 'WITHIN', 'WITHOUT', 'WOMEN', 'WORDS', 'WORKERS', 'WORKING', 'WORLD', 'WORST', 'WOULD', 'WRONG', 'YEARS', 'YESTERDAY', 'YOUNG']

function  genTable(){
//debugger
    let a = `<table><tr>`
    let count = 0
    for (i in words){
        count+=1
        a+=`<td>` + words[i] + `</td>`
        if(count%4==0)
            a+=`</tr><tr>`
    }
    a+=`</tr></table>`
    const l = document.getElementById("words")
    l.innerHTML = a
};

genTable()
// apply some workarounds for opera browser
applyVideoWorkaround();

var player = videojs('myVideo', options, function() {
    // print version information at startup
    var msg = 'Using video.js ' + videojs.VERSION +
        ' with videojs-record ' + videojs.getPluginVersion('record') +
        ' and recordrtc ' + RecordRTC.version;
    videojs.log(msg);
});

// error handling
player.on('deviceError', function() {
    console.warn('device error:', player.deviceErrorCode);
});

player.on('error', function(element, error) {
    console.error(error);
});

// user clicked the record button and started recording
player.on('startRecord', function() {
    console.log('started recording!');
});
// user completed recording and stream is available
player.on('finishRecord', function() {
    console.log('finished recording:', player.recordedData);

    var http = new XMLHttpRequest()
    var data = player.recordedData;
    var serverUrl = 'https://192.168.23.36:8443/upload';
//    var serverUrl = 'http://localhost:8080/upload';
    var formData = new FormData();
    formData.append('file', data, data.name);

    console.log('uploading recording:', data.name);


    $.ajax({
          url: serverUrl,
          type: 'POST',
          data: formData,
          processData: false,
          contentType: false,
          success: function (data) {
            console.log(data);
            const l = document.getElementById("labels")
            l.innerHTML =`<h5>` + data.split(", ").join(", ").slice(0,-3) + `</h5>`
            $("#data-box").removeClass("d-none")
            $("#data-box").addClass("d-flex")
          }
        }).done((data) => {
            console.log('complete')
        })
});
})
