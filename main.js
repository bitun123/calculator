function calculate (value){
    document.getElementById("display").value+=value;
}
function AllDelete (){
    document.getElementById("display").value="";
}
function Delete(){
 const input = document.getElementById("display");
 input.value = input.value.slice(0 , -1)
}
function equal (){
    const display = document.getElementById("display")
    try{
        display.value = eval(display.value)
    }
    catch(error){
        display.value="Error"
    }
}

















