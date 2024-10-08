// Adicionar um evento de envio ao formulário de login
document.getElementById("form").addEventListener("submit", function(event) {
    event.preventDefault(); // Evita que o formulário seja enviado normalmente
    
    var email = document.getElementById("email").value;
    var password = document.getElementById("password").value;
    
    firebase.auth().signInWithEmailAndPassword(email, password)
      .then((userCredential) => {
        // Usuário logado com sucesso
        var user = userCredential.user;
        // Redirecionar o usuário para a página desejada
        window.location.href = "/historico";
      })
      .catch((error) => {
        // Erro ao fazer login
        var errorCode = error.code;
        var errorMessage = error.message;
        // Exibir mensagem de erro para o usuário
        alert("Erro ao fazer login: " + errorMessage);
      });
  });
  