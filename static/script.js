
function responsiveNavScroll() {
  
  var x = document.getElementById("myTopnav");
  
  if (x.className === "topnav") {
    x.className += " responsive";
  } 
  
  else {
    x.className = "topnav";
  }
}

function handleText(){
  var h4 = document.getElementById("h4");
  var textinput = document.getElementById("input");
  h4.innerHTML = textinput.value;

  
}

// call handleText() when the user types in the input element
function init() {
    var input = document.getElementById("input");
    input.addEventListener("keyup", handleText);
    
}

// call init() when the page loads
window.addEventListener("load", init);
